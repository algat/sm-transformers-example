from transformers import Trainer#, nested_detach
from transformers.trainer_pt_utils import nested_detach
from torch.nn import BCEWithLogitsLoss
from torch import nn
from typing import Dict, Union, Any, List, Optional, Tuple
import torch

class MultilabelClassificationTrainer(Trainer):
    # We override the Trainer in order to modify the loss for Multilabel classification
    def compute_loss(self, model, inputs):
        # remove the labels from inputs and compute manually the loss
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return loss
    
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []
            with torch.no_grad():
                if has_labels:
                    labels = inputs.pop("labels")
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                logits = outputs[0]
                if has_labels:
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
                else:
                    loss = None
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

            if prediction_loss_only:
                return (loss, None, None)
            logits = nested_detach(logits)
            if has_labels:
                labels = nested_detach(labels)
            else:
                labels = None
            return (loss, logits, labels)