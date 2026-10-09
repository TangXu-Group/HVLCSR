def create_model(name='hvlcsr', cfg={}):
    if name == 'hvlcsr':
        from hvlcsr_adapter import model_hvlcsr
        return model_hvlcsr(cfg)
    else:
        raise ValueError("模型不支持")
