import pandas as pd


class DataIngestion:

    def __init__(
        self,
        raw_data_path
    ):
        self.raw_data_path = raw_data_path


    def initiate_data_ingestion(self):

        try:

            df = pd.read_csv(
                self.raw_data_path
            )

            return df


        except Exception as e:

            raise Exception(
                f"Error occurred during data ingestion: {e}"
            )
        
        