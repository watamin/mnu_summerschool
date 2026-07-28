from neis_meal_ai.ui import launch_options


def test_colab_launch_uses_share_link() -> None:
    assert launch_options(is_colab=True)["share"] is True


def test_local_launch_stays_local() -> None:
    assert launch_options(is_colab=False)["share"] is False
