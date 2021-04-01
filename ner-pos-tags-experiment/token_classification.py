import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    TokenClassifierOutput)
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertPreTrainedModel)

from experiment_lib.utils import (
    get_label_map,
    get_pos_tag_id_map)

LABEL_LIST, NUM_LABELS, LABEL_TO_ID = get_label_map()
POS_TAG_TO_ID_MAP, POS_ID_TO_TAG_MAP = get_pos_tag_id_map()
POS_TAG_SIZE = len(POS_TAG_TO_ID_MAP.keys())
POS_EMBEDDING_SIZE = 30


class ModifiedBertForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self,
                 config,
                 use_pos_tags=True,
                 freeze_base_model=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_pos_tags = use_pos_tags

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pos_tag_embeddings = nn.Embedding(
            POS_TAG_SIZE, POS_EMBEDDING_SIZE)

        self.classifier = self.build_classifier(
            config=config,
            use_pos_tags=use_pos_tags)

        self.init_weights()

        if freeze_base_model:
            self.freeze_model_parameters()

    def build_classifier(self, config, use_pos_tags):
        if use_pos_tags:
            return nn.Linear(
                config.hidden_size + POS_EMBEDDING_SIZE,
                config.num_labels)
        else:
            return nn.Linear(config.hidden_size, config.num_labels)

    def freeze_model_parameters(self):
        for param in self.bert.base_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pos_tags=None  # TODO: change to IDs...
    ):
        return_dict = self.config.use_return_dict
        if return_dict is not None:
            return_dict = return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # TODO: refactor later...
        if self.use_pos_tags:
            #             pos_tags = torch.tensor(pos_tags).to(torch.long)
            pos_tag_embedding = self.pos_tag_embeddings(pos_tags)
            final_layer_features = torch.cat(
                (sequence_output, pos_tag_embedding), dim=-1)
            logits = self.classifier(final_layer_features)
        else:
            logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(
                        loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
