import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class MultiScaleTemporalCNN(nn.Module):
    def __init__(self, num_classes=5, num_joints=17, in_ch=2, hidden=256, dropout=0.5):
        super().__init__()

        joint_dim    = num_joints * in_ch
        bone_dim     = num_joints * 2
        velocity_dim = num_joints * 2

        def make_embed(in_dim, hidden, dropout):
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.joint_embed    = make_embed(joint_dim,    hidden, dropout)
        self.bone_embed     = make_embed(bone_dim,     hidden, dropout)
        self.velocity_embed = make_embed(velocity_dim, hidden, dropout)

        def make_branches(h, drop):
            return nn.ModuleDict({
                "full": nn.Sequential(TemporalConvBlock(h, h, kernel=3,  dropout=drop)),
                "mid":  nn.Sequential(TemporalConvBlock(h, h, kernel=7,  dropout=drop)),
                "long": nn.Sequential(TemporalConvBlock(h, h, kernel=15, dropout=drop)),
            })

        self.joint_branches    = make_branches(hidden, dropout)
        self.bone_branches     = make_branches(hidden, dropout)
        self.velocity_branches = make_branches(hidden, dropout)

        fused_dim = hidden * 3 * 3
        self.attn_pool = nn.Sequential(
            nn.Linear(fused_dim, 1),
            nn.Softmax(dim=1),
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def _compute_bones(self, x):
        parents = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        bones = x[:, :, :, :2].clone()
        for i, p in enumerate(parents):
            if i != p:
                bones[:, :, i, :] = x[:, :, i, :2] - x[:, :, p, :2]
            else:
                bones[:, :, i, :] = 0.0
        return bones

    def _compute_velocity(self, x):
        xy  = x[:, :, :, :2]
        vel = torch.zeros_like(xy)
        vel[:, 1:, :, :] = xy[:, 1:, :, :] - xy[:, :-1, :, :]
        return vel

    def _forward_stream(self, embed, branches, x_flat, B, T):
        x  = embed(x_flat)
        x  = x.reshape(B, T, -1).permute(0, 2, 1)
        f1 = branches["full"](x)
        f2 = branches["mid"](x)
        f3 = branches["long"](x)
        return torch.cat([f1, f2, f3], dim=1)

    def forward(self, x):
        x = x[..., :2]
        B, T, J, C = x.shape

        feat_joint = self._forward_stream(
            self.joint_embed, self.joint_branches,
            x.reshape(B * T, J * C), B, T,
        )
        bones      = self._compute_bones(x)
        feat_bone  = self._forward_stream(
            self.bone_embed, self.bone_branches,
            bones.reshape(B * T, J * 2), B, T,
        )
        vel        = self._compute_velocity(x)
        feat_vel   = self._forward_stream(
            self.velocity_embed, self.velocity_branches,
            vel.reshape(B * T, J * 2), B, T,
        )

        fused  = torch.cat([feat_joint, feat_bone, feat_vel], dim=1)
        fused  = fused.permute(0, 2, 1)
        attn   = self.attn_pool(fused)
        pooled = (fused * attn).sum(dim=1)
        return self.head(pooled)