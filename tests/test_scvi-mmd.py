from scvi.data import synthetic_iid
from scvi.model import SCVI


def test_scvi_mmd():
    adata = synthetic_iid()
    SCVI.setup_anndata(adata)
    model1 = SCVI(adata)
    model2 = SCVI(adata, mmd=False, beta=1, mode="fast")
    model3 = SCVI(adata, mmd=True, beta=0, mode="normal")
    model4 = SCVI(adata, mmd=True, beta=0.1, mode="fast")

    model1.train()
    model2.train()
    model3.train()
    model4.train()
