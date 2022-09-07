import os
import sys
import flwr as fl
from jsonargparse import ArgumentParser
from jsonargparse.util import usage_and_exit_error_handler
from ..Utils.Helpers.display_helper import Display
from .builder.server_builder import ServerBuilder
from .data_visualization.plot_visualization import ResultsVisualization


class Server():
    def __init__(self, server_address: str, strategy, config) -> None:
        self.server_address = server_address
        self.strategy = strategy
        self.config = config

    def run(self):

        fl.server.start_server(server_address=self.server_address,
                               strategy=self.strategy,
                               config=self.config)

        results_ploter = ResultsVisualization("Z:/Federated Learning Projects/federated-learning-framework/federated_app/centralized_training_result.json")  # noqa: E501
        results_ploter.show_metrics()

def run():
    try:
        parser = ArgumentParser(prog="server_run",
                                error_handler=usage_and_exit_error_handler,
                                description="Start a server for federated tasks.")

        parser.add_argument("--server_configuration_file",
                            "-server",
                            required=True,
                            help="The path of the server configuration file (must be JSON format)")

        program_args = parser.parse_args()

        server_args = ServerBuilder()
        server_args.arg_parse(program_args.server_configuration_file)

        server = Server(server_args.server_address,
                        server_args.strategy,
                        config={"num_rounds": server_args.rounds})
        server.run()
        Display.close()

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()
