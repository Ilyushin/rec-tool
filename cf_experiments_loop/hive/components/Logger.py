import logging

logging.getLogger('thrift').setLevel(logging.CRITICAL)
logging.getLogger('pyhive').setLevel(logging.CRITICAL)
logging.getLogger('elasticsearch').setLevel(logging.CRITICAL)
logging.getLogger('elasticsearch_dsl').setLevel(logging.CRITICAL)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s]: %(message)s',
    level=logging.WARN,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

