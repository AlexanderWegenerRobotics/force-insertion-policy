__all__ = ["GRUBC", "TransformerBC", "build_transformer_model"]


def __getattr__(name):
    if name in __all__:
        from transformer.model import GRUBC, TransformerBC, build_transformer_model

        exports = {
            "GRUBC": GRUBC,
            "TransformerBC": TransformerBC,
            "build_transformer_model": build_transformer_model,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
