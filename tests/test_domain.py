from hifuku.domain import TBRR_SQP_DomainProvider


def test_domain():
    name = TBRR_SQP_DomainProvider.get_domain_name()
    assert name == "TBRR_SQP"
