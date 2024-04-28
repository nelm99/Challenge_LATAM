# log_config.py
import logging
import logging.handlers
import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.TimedRotatingFileHandler('./logs/challenge_latam.log', when='D', interval=1, atTime=datetime.time(23, 59, 59)),
            logging.StreamHandler()
        ]
    )
    logging.info("Se ha reiniciado el Backend. Log configurado correctamente.")