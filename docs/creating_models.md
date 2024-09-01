# Working With Models

In each model folder, there are 3 sub folders:
- `content/`
    - Information about model and performance metrics.
- `docs/`
    - Documentation for the model.
- `src/`
    - Code for the model.

In the `src/` folder, we organize the code into multiple .py files, each serving a specific purpose:
- `methods.py`
    - Containes methods that handle the core logic, such as training and evaluation routines.
- `architecture.py`
    - Defines the model architecture.
- `dataset.py`
    - Handles the dataset and data augmentation definitions.

To run the code and execute these methods, we work from Jupyter notebooks located in the `src/notebooks/` folder. Each notebook is tailored for a specific environment, such as Windows, macOS, Google Colab, and more. This allows you to choose the environment that best suits your needs while leveraging the methods defined in `methods.py`. A tutorial on [how to work with cloud environments](https://github.com/AwrodHaghiTabrizi/UMARV-CV-ScenePerception/blob/main/docs/working_with_environments.md).

For example, while `methods.py` includes the methods to train a model, the actual training process is initiated and managed from a Jupyter notebook specific to your environment in the `src/notebooks/` folder. This setup keeps our code organized and makes it easier to experiment and iterate across different platforms.

## Creating Models

1. Navigate to `src/scripts`.
2. Right click on either `create_model.py` or `create_copy_of_model.py`
    - `create_model.py` creates a new model from the template
    - `create_copy_of_model.py` creates a copy of a model using its model id
3. Click "Run Python File in Termainl".
4. Answer the prompts in the terminal.