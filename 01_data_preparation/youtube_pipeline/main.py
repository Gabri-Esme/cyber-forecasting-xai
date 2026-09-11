import os
from dotenv import load_dotenv
from util import *

def main():
    '''Main function to run the YouTube ETL pipeline'''
    # Load environment variables
    load_dotenv("01_data_preparation/secrets.env")
    api_key = os.getenv("YT_API_KEY")
    db_name = os.getenv("DB_PATH")
    
    # Initialise components
    db = YouTubeDB(db_name)
    api_client = YouTubeClient(api_key)
    
    # ETL Configuration
    etl_config = {
        "countries": COUNTRIES,
        "keywords": KEYWORDS,
        "years": YEARS
        }
       
    # Run ETL
    etl = YouTubeETL(db, api_client, etl_config)
    etl.run()

if __name__ == "__main__":
    # Countries for Query
    COUNTRIES = [
        '(USA|America)',
        '(UK|British|United Kingdom|Britain)',
        '(CANADA|CANADIAN)',
        '(AUSTRALIA)',
        '(Ukraine)',
        '(RUSSIA)',
        '(FRANCE|FRENCH)',
        '(GERMAN)',
        '(Brazil)',
        '(China|chinese)',
        '(Japan)',
        '(Pakistan)',
        '(North Korea)',
        '(South Korea)',
        '(India)',
        '(Taiwan)',
        '(NetherLands|Holland|Dutch)',
        '(SPAIN|Spanish)',
        '(Sweden|Swedish)',
        '(Mexic)',
        '(IRAN)',
        '(ISRAEL)',
        '(Saudi)',
        '(Syria)',
        '(Finland|FINNISH)',
        '(IRELAND|IRISH)',
        '(AUSTRIA)',
        '(NORWAY|Norwegian)',
        '(Switzerland|swiss)',
        '(ITALY|ITALIAN)',
        '(MALAYSIA)',
        '(EGYPT)',
        '(TURKEY|TURKISH)',
        '(portugal|portuguese)',
        '(Palestin|West Bank|GAZA)',
        '(UAE|United Arab Emirates|emarat)']

    # Keywords for Query
    KEYWORDS = ["WAR MILITARY", 
                "WAR ARMED FORCE", 
                "CONFLICT POLITIC", 
                "MILITARY ATTACK", 
                "ARMED FORCE ATTACK"
                ]

    # Time range for Query
    YEARS = [2022, 2023, 2024, 2025, 2026]
    main()
