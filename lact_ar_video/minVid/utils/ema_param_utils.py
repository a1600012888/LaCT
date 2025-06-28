
class EMAParams(object):
    def __init__(self, name_to_trainable_params: dict, ema_weight: float):
        self.ema_weight = ema_weight
        self.name_to_trainable_params = name_to_trainable_params
        self.name_to_ema_params = {}

        for name, param in name_to_trainable_params.items():
            self.name_to_ema_params[name] = param.data.detach().clone()
    

    def update(self):
        for name, param in self.name_to_ema_params.items():
            # check size. to filter out fsdp wrapped params.
            if param.numel() > 0:
                param.data.mul_(self.ema_weight).add_(self.name_to_trainable_params[name].detach(), alpha=1 - self.ema_weight)
            
        
    def copy_to_model(self):
        for name, param in self.name_to_ema_params.items():
            if param.numel() > 0:
                self.name_to_trainable_params[name].data.copy_(param.data)
                
    def copy_from_model(self):
        for name, param in self.name_to_ema_params.items():
            if param.numel() > 0:
                self.name_to_ema_params[name].data.copy_(self.name_to_trainable_params[name].data.detach())
            
    
    def cache_model(self, cpu=False):
        """
        When checkpoint the model, we need to first
        cache_model()
        copy_to_model()
        then save the model weights. 
        restore_model_from_cache()

        This way, the ema params could be made compatible with the both DDP, and FSDP and paratially trained models. 
        """
        cache_dict = {}
        if cpu:
            for name, param in self.name_to_trainable_params.items():
                cache_dict[name] = param.data.detach().cpu().clone()
        else:
            for name, param in self.name_to_trainable_params.items():
                cache_dict[name] = param.data.detach().clone()
        
        self.cache_dict = cache_dict
    
    def restore_model_from_cache(self):
        for name, param in self.name_to_trainable_params.items():
            param.data.copy_(self.cache_dict[name].to(param.device))
        
        # then clear the cache
        self.cache_dict = None
