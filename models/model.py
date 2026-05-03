from torch_geometric.nn import GCNConv
from torch import nn
import torch
import torch.nn.functional as F


class MultiLayerGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.3):
        super().__init__()
        # Add numerical stability processing
        self.eps = 1e-5
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight):
        try:
            # Check if input contains NaN or Inf
            assert not torch.isnan(x).any(), "\033[33mInput x contains NaN.\033[0m"
            assert not torch.isnan(edge_weight).any(), "\033[33mEdge weights contain NaN\033[0m"
            assert not torch.isinf(x).any(), "\033[33mInput x contains Inf\033[0m"
            assert not torch.isinf(edge_weight).any(), "\033[33mEdge weights contain Inf\033[0m"

            # Numerical stability processing
            edge_weight = F.softmax(edge_weight, dim=0)  # Normalize edge weights

            # Project input to unified dimensio
            x = self.input_proj(x)

            # Residual connection + multi-layer GCN
            for i, (gcn, norm, batch_norm) in enumerate(zip(self.gcn_layers, self.norms, self.batch_norms)):
                identity = x

                # GCN layer
                x = gcn(x, edge_index, edge_weight)

                # Normalization
                x = batch_norm(x)
                x = F.gelu(x)
                x = norm(x)

                # Dropout
                x = self.dropout(x)

                # Residual connection
                x = x + identity

                # Numerical stability check
                if torch.isnan(x).any():
                    print(f"\033[33mNaN detected in layer {i}\033[0m")
                    return None

            return x

        except Exception as e:
            print(f"\n\033[33mError in MultiLayerGCN:\033[0m")
            print(f"\033[33mError message: {str(e)}\033[0m")
            print(f"\033[33mLayer index: {i if 'i' in locals() else 'Not reached layers'}\033[0m")
            if 'x' in locals():
                print(f"\033[33mCurrent x stats:\033[0m")
                print(f"\033[33m- Shape: {x.shape}\033[0m")
                print(f"\033[33m- Contains NaN: {torch.isnan(x).any()}\033[0m")
                print(f"\033[33m- Contains Inf: {torch.isinf(x).any()}\033[0m")
            raise e

class FeatureFusion(nn.Module):
    def __init__(self, temporal_dim, spatial_dim, static_dim, ndvi_dim):
        super().__init__()
        self.num_stations = 6

        # Record input dimensions
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.static_dim = static_dim
        self.ndvi_dim = ndvi_dim

        # Feature dimension unification
        self.temporal_proj = nn.Linear(temporal_dim, spatial_dim)
        self.static_proj = nn.Linear(static_dim, spatial_dim)
        self.ndvi_proj = nn.Linear(ndvi_dim, spatial_dim)

        self.temporal_attention = nn.MultiheadAttention(embed_dim=spatial_dim, num_heads=2, batch_first=True)
        self.spatial_attention = nn.MultiheadAttention(embed_dim=spatial_dim, num_heads=2, batch_first=True)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(spatial_dim)
        self.norm2 = nn.LayerNorm(spatial_dim)
        self.norm3 = nn.LayerNorm(spatial_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(spatial_dim * 4, spatial_dim * 4),
            nn.LayerNorm(spatial_dim * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(spatial_dim * 4, spatial_dim * 2),
            nn.LayerNorm(spatial_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(spatial_dim * 2, spatial_dim),
            nn.LayerNorm(spatial_dim)
        )

        self.feature_gates = nn.Parameter(torch.ones(4, spatial_dim) * 0.5)
        
        self.skip_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, features):
        batch_size = features.size(0) // self.num_stations

        # Split features
        start_idx = 0
        temporal_features = features[:, start_idx:start_idx + self.temporal_dim]
        start_idx += self.temporal_dim
        spatial_features = features[:, start_idx:start_idx + self.spatial_dim]
        start_idx += self.spatial_dim
        static_features = features[:, start_idx:start_idx + self.static_dim]
        start_idx += self.static_dim
        ndvi_features = features[:, start_idx:start_idx + self.ndvi_dim]

        # Reshape features
        temporal_features = temporal_features.view(batch_size, self.num_stations, -1)
        spatial_features = spatial_features.view(batch_size, self.num_stations, -1)
        static_features = static_features.view(batch_size, self.num_stations, -1)
        ndvi_features = ndvi_features.view(batch_size, self.num_stations, -1)

        # Project to unified dimension
        temporal_proj = self.temporal_proj(temporal_features)
        spatial_proj = spatial_features
        static_proj = self.static_proj(static_features)
        ndvi_proj = self.ndvi_proj(ndvi_features)

        # Apply attention mechanism
        temp_attn, _ = self.temporal_attention(temporal_proj, temporal_proj, temporal_proj)
        temp_attn = self.norm1(temp_attn + temporal_proj)

        spat_attn, _ = self.spatial_attention(spatial_proj, spatial_proj, spatial_proj)
        spat_attn = self.norm2(spat_attn + spatial_proj)

        gates = torch.sigmoid(self.feature_gates)

        combined = torch.cat([
            temp_attn * gates[0],
            spat_attn * gates[1],
            static_proj * gates[2],
            ndvi_proj * gates[3]
        ], dim=-1)

        output = self.fusion_mlp(combined)
        
        skip_value = temporal_proj.mean(dim=1)
        output = output + self.skip_weight * skip_value.unsqueeze(1).expand_as(output)
        
        output = self.norm3(output)

        return output

class STGCN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, num_stations=6, dropout=0.5, 
                 window_size=144, prediction_length=10, temp_loss_weight=0.05):
        super(STGCN, self).__init__()
        self.num_stations = num_stations
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim
        self.window_size = window_size
        self.num_barrier_types = 3  # Number of barrier types

        # 1. Static feature processing layer
        self.static_encoder = nn.Sequential(
            nn.Linear(self.num_barrier_types, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. NDVI processing layer
        self.ndvi_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. Temporal feature extraction
        self.temporal_gru = nn.GRU(
            input_size=3,  # Meteorological feature dimension
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 4. Spatial relation encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 5. Graph convolution layer
        self.multi_gcn = MultiLayerGCN(hidden_dim * 2, hidden_dim)

        # 6. Feature fusion layer
        self.feature_fusion = FeatureFusion(
            temporal_dim=2 * hidden_dim,
            spatial_dim=hidden_dim,
            static_dim=hidden_dim,
            ndvi_dim=hidden_dim
        )

        # 7. Prediction layer (enhanced)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prediction_length * output_dim)
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(hidden_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # Add edge weight projection layer
        self.edge_weight_proj = nn.Linear(hidden_dim, 1)

        self.feature_weights = nn.Parameter(torch.tensor([1.5, 1.2, 1.2]))
        self.temp_loss_weight = temp_loss_weight
        self.prediction_length = prediction_length
        
        self.register_buffer('weight_ema', torch.zeros(3))
        self.register_buffer('error_history', torch.zeros(3))
        self.register_buffer('error_std', torch.ones(3))
        self.error_momentum = 0.98
        self.ema_momentum = 0.95
        self.update_counter = 0
        
        # weight control parameters
        self.min_weight = 0.2
        self.max_weight = 2.0
        self.weight_sum = 3.0

        # loss function related parameters
        self.l1_weight = 0.0001  # L1 regularization weight
        self.l2_weight = 0.001  # L2 regularization weight
        self.grad_penalty_weight = 0.001  #  Gradient penalty weight
        self.feature_smoothness_weight = 0.0005  # Feature smoothness weight
        
        #  Feature importance weights (based on physical meaning)
        self.feature_importance = nn.Parameter(torch.tensor([1.2, 1.0, 1.0]))
        
        #  Initialize error 
        self.register_buffer('error_ema', torch.zeros(3))

    def get_feature_errors(self, pred, target):
        """
        Calculate prediction error for each feature
        """
        errors = (pred - target) ** 2
        feature_errors = torch.mean(errors, dim=0)
        
        target_std = torch.std(target, dim=0) + 1e-6
        feature_relative_errors = feature_errors / target_std
        
        return feature_errors, feature_relative_errors

    def update_feature_weights(self, feature_errors):
        """
        More conservative weight update strategy
        """
        with torch.no_grad():
            if self.update_counter == 0:
                self.error_history = feature_errors
                self.error_std = torch.std(feature_errors.unsqueeze(0))
            else:
                self.error_history = 0.8 * self.error_history + 0.2 * feature_errors
                new_std = torch.std(torch.stack([self.error_history, feature_errors]), dim=0)
                self.error_std = 0.8 * self.error_std + 0.2 * new_std
            
            self.update_counter += 1
            
            mean_error = torch.mean(feature_errors)
            relative_errors = feature_errors / (mean_error + 1e-6)
            
            # Update relative error using moving average
            if not hasattr(self, 'relative_errors_ma'):
                self.relative_errors_ma = relative_errors
            else:
                self.relative_errors_ma = 0.7 * self.relative_errors_ma + 0.3 * relative_errors

            # Calculate target weights
            # Use squared inverse relationship instead of simple inverse, so high error features decay faster
            target_weights = 1.0 / ((self.relative_errors_ma ** 1.5) + 1e-5)

            # Fuse feature physical importance
            importance_factor = F.softmax(self.feature_importance, dim=0)
            target_weights = target_weights * importance_factor

            target_weights = target_weights / target_weights.sum() * self.weight_sum
            lr = 0.01

            # Adaptive learning rate: adjust based on training progress
            if self.update_counter > 50:
                progress_factor = min(1.0, self.update_counter / 150)
                lr = max(0.005, 0.01 * (1.0 - 0.7 * progress_factor))
            
            # Calculate weight update
            current_weights = self.feature_weights.data
            weight_update = lr * (target_weights - current_weights)
            
            # Limit single update magnitude
            max_update = 0.1
            weight_update = torch.clamp(weight_update, -max_update, max_update)
            
            new_weights = current_weights + weight_update
            
            #  Ensure weights are within reasonable range
            new_weights = torch.clamp(new_weights, self.min_weight, self.max_weight)
            
            weight_sum = torch.sum(new_weights)
            if weight_sum > 0:
                new_weights = new_weights * (self.weight_sum / weight_sum)
            
            self.feature_weights.data = 0.8 * new_weights + 0.2 * current_weights
            

    def compute_gradient_penalty(self, pred, target):
        """gradient penalty calculation"""
        try:
            # Calculate MSE loss between predictions and targets
            loss = F.mse_loss(pred, target)
            
            # gradients
            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=pred,
                grad_outputs=torch.ones_like(loss),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
            
            #  gradient penalty
            gradient_penalty = ((gradients_norm - 1) ** 2).mean()
            
            return gradient_penalty
        
        except Exception as e:
            print(f"\033[33mError in gradient penalty computation: {str(e)}\033[0m")
            return torch.zeros(1, device=pred.device, requires_grad=True)

    def feature_smoothness_loss(self, pred):
        """Calculate smoothness loss between features"""
        pred = pred.view(-1, self.prediction_length, 3)
        feature_diff = torch.abs(pred[..., 1:] - pred[..., :-1])
        return torch.mean(feature_diff)

    def adaptive_feature_weights(self, feature_errors):
        """"Improved adaptive feature weight calculation"""
        with torch.no_grad():
            # Update error EMA
            self.error_ema = 0.95 * self.error_ema + 0.05 * feature_errors
            
            # Calculate normalized error
            normalized_errors = self.error_ema / (torch.mean(self.error_ema) + 1e-6)
            
            importance_factor = F.softmax(self.feature_importance, dim=0)
            error_factor = 1.0 / (normalized_errors + 1e-6)
            
            # Calculate final weights
            weights = importance_factor * error_factor
            weights = F.softmax(weights, dim=0)
            
            return weights

    def weighted_mse_loss(self, pred, target):
        """Optimized weighted MSE loss"""

        feature_errors, feature_relative_errors = self.get_feature_errors(pred, target)
        
        # Calculate MSE loss for each feature
        mse_losses = torch.mean((pred - target) ** 2, dim=0)  # [4]
        
        # Apply adaptive weights
        adaptive_weights = self.adaptive_feature_weights(feature_errors)
        weighted_mse = torch.sum(mse_losses * adaptive_weights)
        
        # L1 regularization 
        l1_reg = self.l1_weight * torch.mean(torch.abs(self.feature_weights - 1.0))
        
        # L2 regularization 
        l2_reg = self.l2_weight * torch.mean(self.feature_weights ** 2)
        
        # Feature smoothness loss
        smoothness = self.feature_smoothness_weight * self.feature_smoothness_loss(pred)
        
        # Total loss
        total_loss = (
            weighted_mse + 
            l1_reg + 
            l2_reg * 2.0 +
            smoothness
        )
        
        # Update feature weights
        if self.training:
            self.update_feature_weights(feature_errors)
        
        return total_loss, feature_errors, feature_relative_errors

    def temporal_consistency_loss(self, pred):
        """Simplified temporal continuity loss"""
        pred = pred.view(-1, self.prediction_length, 3)
        
        # Only keep short-term continuity loss
        diff = pred[:, 1:] - pred[:, :-1]
        loss = torch.mean(diff ** 2)
        
        return loss

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        device = x.device
        x = x.float().to(device)
        edge_attr = edge_attr.float().to(device)
        edge_index = edge_index.to(device)

        batch_size = 1 if not hasattr(data, 'batch') else (data.batch[-1].item() + 1)
        time_steps = x.size(0) // (batch_size * self.num_stations)

        try:
            # Reshape input data
            x = x.view(batch_size, self.num_stations, time_steps, -1)

            # Separate features
            num_dynamic_features = 4  # temperature, humidity, pm10, ndvi
            num_barrier_types = 3     # Number of barrier types
            num_time_features = 6

            # 1. Separate features
            climate_data = x[..., :3]  # temperature, humidity, pm10
 
            ndvi_data = x[..., 3:4]   # ndvi
            barrier_data = x[..., 4:4+num_barrier_types]  # sand_barrier one-hot
            time_features = x[..., -num_time_features:]   # time encoding

            # 2. Processing time series features
            temporal_features = []
            for i in range(self.num_stations):
                station_seq = climate_data[:, i, :, :]
                out, _ = self.temporal_gru(station_seq)
                temporal_features.append(out[:, -1, :])

            temporal_features = torch.stack(temporal_features, dim=1)

            # 3. Processing NDVI feature
            ndvi_features = []
            for i in range(self.num_stations):
                station_ndvi = ndvi_data[:, i, :, :]
                ndvi_seq = station_ndvi.reshape(batch_size * time_steps, 1)
                ndvi_encoded = self.ndvi_encoder(ndvi_seq)
                ndvi_encoded = ndvi_encoded.view(batch_size, time_steps, -1)
                ndvi_features.append(ndvi_encoded[:, -1, :])

            ndvi_features = torch.stack(ndvi_features, dim=1)

            # 4. Processing sand-barrier type features
            barrier_features = barrier_data[:, :, -1, :]
            barrier_features = barrier_features.view(batch_size * self.num_stations, -1)
            barrier_features = self.static_encoder(barrier_features)


            # 5. Processing spatial relationships
            batch_edge_index = []
            batch_edge_attr = []
            base_edge_index = edge_index[:, :30]
            base_edge_attr = edge_attr[:30]

            for b in range(batch_size):
                current_edge_index = base_edge_index.clone()
                current_edge_index = current_edge_index + (b * self.num_stations)
                batch_edge_index.append(current_edge_index)
                batch_edge_attr.append(base_edge_attr)

            edge_index = torch.cat(batch_edge_index, dim=1)
            edge_attr = torch.cat(batch_edge_attr, dim=0)

            # 6. Processing edge weights
            edge_weights = self.spatial_encoder(edge_attr)
            edge_weights = self.edge_weight_proj(edge_weights).squeeze(-1)

            # 7. Feature fusion
            temporal_features = temporal_features.view(batch_size * self.num_stations, -1)
            spatial_features = self.multi_gcn(temporal_features, edge_index, edge_weights)
            barrier_features = barrier_features.view(batch_size * self.num_stations, -1)
            ndvi_features = ndvi_features.view(batch_size * self.num_stations, -1)

            # 8. Merge all features
            fused_features = torch.cat([
                temporal_features,
                spatial_features,
                barrier_features,
                ndvi_features
            ], dim=-1)

            # 9. Using feature moudule
            fused_features = self.feature_fusion(fused_features)

            # 10. Generating predictions
            predictions = self.predictor(fused_features)
            predictions = predictions.view(-1, 3)

            # loss
            total_loss, feature_errors, feature_relative_errors = self.weighted_mse_loss(predictions, data.y)
            temp_loss = self.temporal_consistency_loss(predictions)
            
            # total loss
            total_loss = total_loss + self.temp_loss_weight * temp_loss

            return predictions, total_loss, feature_errors, feature_relative_errors

        except Exception as e:
            print(f"\033[33m\nError in forward pass:\033[0m")
            print(f"\033[33mError message: {str(e)}\033[0m")
            print(f"\033[33mAdditional debug info:\033[0m")
            print(f"\033[33mInput shape: {x.shape}\033[0m")
            print(f"\033[33mBatch size: {batch_size}\033[0m")
            print(f"\033[33mTime steps: {time_steps}\033[0m")
            print(f"\033[33mBarrier data shape: {barrier_data.shape if 'barrier_data' in locals() else 'Not created'}\033[0m")
            print(f"\033[33mBarrier features shape before encoding: {barrier_features.shape if 'barrier_features' in locals() else 'Not created'}\033[0m")
            raise e
