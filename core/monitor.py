import json
from core.occurence import Occurence


class Monitor(object):
    def __init__(self, simulation):
        self.simulation = simulation
        self.occurence_monitor = simulation.occurence_monitor
        self.event_file = simulation.event_file
        self.states = []

    def run(self):
        self.state_record()

    def state_record(self):
        state = {
            'timestamp': self.occurence_monitor.now,
            'cluster_state': self.simulation.cluster.state
        }
        self.states.append(state)
        if not self.simulation.finished:
            state_record_occurence = Occurence(
                    trigger_time=self.occurence_monitor.now + 1,
                    otype = Occurence.RECORD,
                    action=self.state_record
                )
            self.occurence_monitor.add_occurence(state_record_occurence)   
        else:
            self.write_to_file()         

    def write_to_file(self):
        with open(self.event_file, 'w') as f:
            json.dump(self.states, f, indent=4)
