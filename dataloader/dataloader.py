import tensorflow_datasets as tfds


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        return tfds.load(
            name=data_config.ds_name,
            split=data_config.split,
            data_dir=data_config.ds_path,
            shuffle_files=data_config.shuffle_files,
            download=True,
            as_supervised=data_config.as_supervised,
            with_info=data_config.load_with_info,
        )