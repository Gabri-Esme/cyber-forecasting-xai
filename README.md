This is a Python Implementation of the software introduced in the MSc Report "Explainability and Reliance in Cyber Threat Forecasting".

Suggested order:
1)	Install requirements
2)	Open “01_data_prepartion” and insert a file named “secrets.env”. You will need to obtain an API key for YouTube and Elsevier, as well as assign a path to the database. Assign them to "YT_API_KEY", "ELS_API_KEY" and "DB_PATH" respectively. For MSc markers, keys have been provided in the report and you are not required to obtain those keys.
3)	"01_data_preparation/elsevier_data/main.py” > RUN
4)	“01_data_preparation/holidays/py_holiday_counter.py” > RUN
5)	“01_data_preparation/youtube_pipeline/main.py” > RUN (This will eventually stop when rate is exceeded)
6)	“01_data_preparation/incident_data/transformer.py” > RUN
7)	OPTIONAL: “01_data_preparation/incident_data/gaussian_clusters/gaussian_stats.ipynb”
8)	“01_data_preparation/compile_dataset.py” > RUN
9)	Open Terminal
11)	Navigate to the directory “… /cyber-forecasting-xai”
11)	Run the following command:
python 02_model/main.py --mode test_hps --iterations 100 --epochs 200
(Please be advised this step takes multiple hours)

12)	Run the following command:
python 02_model/main.py --mode train

13)	Run the following command:
python 02_model/main.py --mode forecast

14)	OPTIONAL: Run the following command:
python 02_model/main.py --mode explain


