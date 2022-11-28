import os
import pytest


def get_files(path):
    ignore_list = ["__pycache__"]
    path = os.path.abspath(path)
    files = []
    for file in os.listdir(path):
        if file not in ignore_list:
            files.append(os.path.join(path, file))
    return files


@pytest.mark.parametrize("example", get_files("examples/gym"))
def test_gym_examples(example):
    result = os.system("python" + " " + example + " " + "--max_backprops 1")
    success = result == 0
    assert success is True


@pytest.mark.parametrize("example", get_files("examples/unity"))
def test_unity_examples(example):
    result = os.system("python" + " " + example + " " + "--max_backprops 1")
    success = result == 0
    assert success is True


@pytest.mark.parametrize("example", get_files("examples/tutorials"))
def test_tutorial_examples(example):
    result = os.system("python" + " " + example + " " + "--max_backprops 1")
    success = result == 0
    assert success is True


@pytest.mark.parametrize("example", get_files("examples/brax"))
def test_brax_examples(example):
    result = os.system("python" + " " + example + " " + "--max_backprops 1")
    success = result == 0
    assert success is True
