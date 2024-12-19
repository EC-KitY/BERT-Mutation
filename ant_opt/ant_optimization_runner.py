from eckity.evaluators import SimpleIndividualEvaluator
from ant_opt.artifial_ant_problem import AntSimulator, progn, prog2, prog3, if_then_else


class ArtificialAntEvaluator(SimpleIndividualEvaluator):
    def __init__(self, ant_instance):
        super().__init__()

        self.ant = ant_instance

    def evaluate_individual(self, individual):
        self.ant.run(individual)
        return self.ant.eaten


ant_sim = AntSimulator(600)
ant_evaluator = ArtificialAntEvaluator(ant_sim)

function_set = [
    prog2,
    prog3,
    if_then_else,
    ant_sim.turn_left,
    ant_sim.turn_right,
    ant_sim.move_forward,
]