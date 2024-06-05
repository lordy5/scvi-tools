from scvi.data import synthetic_iid
from scvi.model import SCVI


def test_scvi_mmd():
    # initialization for different combinations of paramters
    adata = synthetic_iid()
    SCVI.setup_anndata(adata)
    model1 = SCVI(adata)
    model2 = SCVI(adata, mmd=False, beta=1, mode="fast")
    model3 = SCVI(adata, mmd=True, beta=0, mode="normal")
    model4 = SCVI(adata, mmd=True, beta=0.1, mode="fast")

    # training
    model1.train(max_epochs=5)
    model2.train(max_epochs=5)
    model3.train(max_epochs=5)
    model4.train(max_epochs=5)

    # inference
    model1.get_latent_representation()
    model2.get_latent_representation()
    model3.get_latent_representation()
    model4.get_latent_representation()

    # check if mmd loss is recorded
    assert len(model1.history["mmd"]) == 5
    assert len(model2.history["mmd"]) == 5
    assert len(model3.history["mmd"]) == 5
    assert len(model4.history["mmd"]) == 5
