import pytest
import arxiv_lib.config as config
import arxiv_lib.ingest as ingest
from arxiv_lib.appdb import init_app_db
from arxiv_lib.config import APP_DB_PATH


@pytest.fixture()
def data_dir(tmp_path):
    """Redirect all config path functions to a fresh temporary directory."""
    original = config.DATA_DIR
    config.DATA_DIR = str(tmp_path)
    ingest._embedding_db_initialized = False
    yield tmp_path
    config.DATA_DIR = original
    ingest._embedding_db_initialized = False


@pytest.fixture()
def app_db_con(data_dir):
    """Initialise app.db schema in the isolated data dir and yield an open connection."""
    init_app_db(APP_DB_PATH())
    from arxiv_lib.appdb import get_connection
    con = get_connection(APP_DB_PATH())
    yield con
    con.close()
