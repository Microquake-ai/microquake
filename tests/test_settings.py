

# import os
# def test_common_directory(settings):
#     common_dir = os.path.join(os.getcwd() + '/common')
#     assert common_dir == settings.common_dir


def test_project_code(settings):
    assert 'OT' == settings.PROJECT_CODE
