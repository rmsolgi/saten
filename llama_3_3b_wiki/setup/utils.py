from lm_eval.models.huggingface import HFLM

class DummyCacheHook:
    def add_partial(self, *args, **kwargs):
        pass

class InMemoryLlama(HFLM):
    def __init__(self, the_model, tokenizer, batch_size=1, max_length=4096):
        self._model = the_model
        self.tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = the_model.device
        self._config = the_model.config  # no need to modify if you just replaced layers
        self._rank = 0
        self._world_size = 1  
        self.backend = "causal"
        self.add_bos_token = False
        self.logits_cache = None
        self.batch_size_per_gpu = 1
        self.cache_hook = DummyCacheHook()
        self.revision = None
        self.pretrained = None
        self.peft = False
        self.delta = None
        self.custom_prefix_token_id = None

        # NOTE: don't call super().__init__() — bypass pretrained loading

    def _model_call(self, inputs, attention_mask=None):
        outputs = self._model(input_ids=inputs, attention_mask=attention_mask)
        return outputs.logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self._model.generate(
            input_ids=context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )