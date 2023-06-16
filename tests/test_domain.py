from hifuku.domain import TORR_SQP_Domain


def test_domain():
    assert TORR_SQP_Domain.get_domain_name() == "TORR_SQP"
