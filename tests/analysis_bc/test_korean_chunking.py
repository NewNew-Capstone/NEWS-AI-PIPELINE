from analysis_bc.preprocessor import split_into_sentences


def test_long_korean_text_splits_on_transition_phrase() -> None:
    prefix = (
        "\uc815\ubd80\ub294 \uc624\ub298 \uc624\uc804 \uad00\uacc4 \ubd80\ucc98 \ud569\ub3d9 "
        "\ube0c\ub9ac\ud551\uc5d0\uc11c \uc0c8\ub85c\uc6b4 \uc815\ucc45\uc758 \ucd94\uc9c4 "
        "\ubc30\uacbd\uacfc \uc138\ubd80 \uc77c\uc815 \uadf8\ub9ac\uace0 \uc608\uc0b0 "
        "\uc9d1\ud589 \uacc4\ud68d\uc744 \ucc28\ub840\ub85c \uc124\uba85\ud588\ub2e4"
    )
    suffix = (
        "\ud558\uc9c0\ub9cc \uc774\uac74 \ub108\ubb34 \uc131\uae09\ud55c "
        "\uacb0\uc815\uc774\ub77c\ub294 \ube44\ud310\ub3c4 \ub098\uc624\uace0 "
        "\uc2dc\ubbfc\ub4e4 \uc785\uc7a5\uc5d0\uc11c\ub294 \ubd88\uc548\uacfc "
        "\ubd84\ub178\uac00 \ucee4\uc9c8 \uc218 \uc788\ub2e4\ub294 "
        "\uc9c0\uc801\uc774 \uc774\uc5b4\uc9c0\uace0 \uc788\ub2e4"
    )

    result = split_into_sentences(f"{prefix} {suffix}", "ko")

    assert [s.content_sentence_id for s in result] == [0, 1]
    assert [s.sentence_order for s in result] == [0, 1]
    assert result[0].sentence_text == prefix
    assert result[1].sentence_text == suffix


def test_short_korean_sentence_stays_single_chunk() -> None:
    text = "\uc815\ubd80\ub294 \uc624\ub298 \uc0c8 \uc815\ucc45\uc744 \ubc1c\ud45c\ud588\ub2e4"

    result = split_into_sentences(text, "ko")

    assert len(result) == 1
    assert result[0].sentence_text == text


def test_commas_alone_do_not_over_split() -> None:
    text = (
        "\uc815\ubd80\ub294 \uc608\uc0b0\uacfc \uc77c\uc815, \ub300\uc0c1 \uc9c0\uc5ed\uacfc "
        "\uc2e0\uccad \ubc29\uc2dd, \ud604\uc7a5 \uc810\uac80 \uacc4\ud68d\uc744 "
        "\uc124\uba85\ud588\uace0 \uad00\ub828 \uc790\ub8cc\ub294 \uc624\ud6c4\uc5d0 "
        "\uacf5\uac1c\ud55c\ub2e4\uace0 \ubc1d\ud614\ub2e4"
    )

    result = split_into_sentences(text, "ko")

    assert len(result) == 1
    assert result[0].sentence_text == text


def test_long_korean_text_falls_back_to_length_split() -> None:
    text = (
        "\uc815\ubd80\ub294 \uc624\ub298 \uc624\uc804 \uad00\uacc4 \ubd80\ucc98 \ud569\ub3d9 "
        "\ube0c\ub9ac\ud551\uc5d0\uc11c \uc0c8\ub85c\uc6b4 \uc815\ucc45\uc758 \ucd94\uc9c4 "
        "\ubc30\uacbd\uacfc \uc138\ubd80 \uc77c\uc815 \uadf8\ub9ac\uace0 \uc608\uc0b0 "
        "\uc9d1\ud589 \uacc4\ud68d\uc744 \ucc28\ub840\ub85c \uc124\uba85\ud588\uace0 "
        "\ud604\uc7a5 \ub2f4\ub2f9\uc790\ub4e4\uc740 \ub2e4\uc74c \uc8fc\ubd80\ud130 "
        "\uc2e0\uccad \uc808\ucc28\uc640 \ubb38\uc758 \ucc3d\uad6c\ub97c \uc21c\ucc28\uc801\uc73c\ub85c "
        "\uc548\ub0b4\ud560 \uc608\uc815\uc774\ub77c\uace0 \uc124\uba85\ud588\uace0 "
        "\ucd94\uac00 \uc138\ubd80 \uae30\uc900\uc740 \ubcc4\ub3c4 \uc790\ub8cc\ub85c "
        "\uacf5\uac1c\ud55c\ub2e4\uace0 \uc124\uba85\ud588\ub2e4"
    )

    result = split_into_sentences(text, "ko")

    assert len(result) == 2
    assert [s.content_sentence_id for s in result] == [0, 1]
    assert all(len(s.sentence_text) >= 20 for s in result)
