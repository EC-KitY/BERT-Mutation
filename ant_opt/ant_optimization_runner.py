from eckity.algorithms import SimpleEvolution
from eckity.base.utils import arity
from eckity.creators import HalfCreator
from eckity.evaluators import SimpleIndividualEvaluator
from eckity.genetic_operators import SubtreeCrossover, SubtreeMutation, TournamentSelection
from eckity.statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation

from ant_opt.artifial_ant_problem import AntSimulator, prog2, prog3
from uniform_mutation import UniformNodeMutation

"""
Questions:
1. I don't need float terminals, so how can I remove them?
2. I need to compile each individual into python code and run it. How can I do that?
3. the arity(func) is problematic when used with classes 
"""
class ArtificialAntEvaluator(SimpleIndividualEvaluator):
    def __init__(self, ant_instance):
        super().__init__()

        self.ant = ant_instance

    def evaluate_individual(self, individual):
        # todo: somehow execute the individual
        # individual.execute()
        # self.ant.run(individual)
        print(individual)
        return self.ant.eaten


ant_sim = AntSimulator(600)

with open("./santafe_trail.txt") as trail_file:
    ant_sim.parse_matrix(trail_file)

ant_evaluator = ArtificialAntEvaluator(ant_sim)


def if_food_ahead(out1, out2):
    return ant_sim.if_food_ahead(out1, out2)


def turn_left():
    ant_sim.turn_left()


def turn_right():
    ant_sim.turn_right()


def move_forward():
    ant_sim.move_forward()


function_set = [
    prog2,
    prog3,
    if_food_ahead
]

terminal_set = [
    turn_left,
    turn_right,
    move_forward,
]

function_arities = [
    arity(f) for f in function_set
]

algo = SimpleEvolution(
    Subpopulation(
        creators=HalfCreator(
            init_depth=(3, 6),
            terminal_set=terminal_set,
            function_set=function_set,
            erc_range=None,
            bloat_weight=0.0001,
        ),
        population_size=100,
        # user-defined fitness evaluation method
        evaluator=ant_evaluator,
        # minimization problem (fitness is MAE), so higher fitness is worse
        higher_is_better=False,
        elitism_rate=0.05,
        # genetic operators sequence to be applied in each generation
        operators_sequence=[
            SubtreeCrossover(probability=0.9),
            SubtreeMutation(probability=0.2),
            UniformNodeMutation(node_probability=0.1),
        ],
        selection_methods=[
            # (selection method, selection probability) tuple
            (
                TournamentSelection(
                    tournament_size=4
                ),
                1,
            )
        ],
    ),
    max_workers=1,
    max_generation=40,
    termination_checker=None,
    statistics=BestAverageWorstStatistics(),
)

# evolve the generated initial population
algo.evolve()

# execute the best individual after the evolution process ends, by assigning numeric values to the variable
# terminals in the tree
# print(f"algo.execute(x=2,y=3,z=4): {algo.execute(x=2, y=3, z=4)}")
