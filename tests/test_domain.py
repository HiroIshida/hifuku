from hifuku.domain import TBRR_SQP_Domain


def test_domain():
    assert TBRR_SQP_Domain.get_domain_name() == "TBRR_SQP"
