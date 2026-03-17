class BlackJaxNestedSampling:

    def __init__(self,
                 likelihood,
                 prior,
                 rng_key: PRNGKey):
            
        
        raise NotImplementedError(f"blackjax nested sampling still needs to be implemented.")
    
    def sample(self, key: PRNGKey):
        raise NotImplementedError
    
    def save(self, sampler_extra_output: bool, outdir: str) -> None:
        raise NotImplementedError
    
    def print_summary(self,):
        raise NotImplementedError