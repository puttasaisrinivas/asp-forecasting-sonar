class ConfigDetails:
    def __init__(self, data_dict: dict):
        """
        Initializes the ConfigDetails object.

        Args:
            data_dict (dict): Dictionary containing configuration details.
        """
        self.master_data = data_dict["master_data"]
        self.asp_data = data_dict["asp_data"]
        self.ep_data = data_dict["ep_data"]
        self.forecast_months = data_dict["forecast_months"]
        self.lst_prioritized_jcodes = list(
            data_dict["lst_prioritized_products"]["J_CODE"].unique()
        )
        self.lst_available_jcodes = list(
            data_dict["lst_selected_products"]["J_CODE"].unique()
        )
        self.files_location = data_dict["files_location"]
