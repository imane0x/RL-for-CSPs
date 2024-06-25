import torch.nn.functional as F
import torch.nn as nn
class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.predict_action = nn.Linear(in_features=128, out_features=100, bias=True)
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        act_dim =1
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_pred = torch.argmax(action_preds, dim=1, keepdim=True)
        action_pred = action_pred.float()
        action_pred.requires_grad = True

        loss = F.cross_entropy(action_preds, torch.squeeze(action_targets).long())


        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)
