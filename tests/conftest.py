def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda:0")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.device
    if 'device' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("device", [option_value], scope='module')