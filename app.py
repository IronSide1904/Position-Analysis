from data_sources.runtime_env import prepare_external_data_env

prepare_external_data_env()

from ui.dashboard_v2 import render_dashboard


if __name__ == "__main__":
    render_dashboard()
