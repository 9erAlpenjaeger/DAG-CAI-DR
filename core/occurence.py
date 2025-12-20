import heapq
from datetime import datetime


class Occurence():
    # 
    ARRIVAL = 0
    FINISH = 1
    PROVISION = 2
    RECORD = 3
    def __init__(self, trigger_time, otype, action, *args, **kwargs):
        self.trigger_time = trigger_time
        self.otype = otype # 
        self.action = action
        self.args = args
        self.kwargs = kwargs

        self.timeleft = None

    def __lt__(self, other):
        if self.trigger_time < other.trigger_time:
            return True  
        elif self.trigger_time > other.trigger_time:
            return False   
        else:
            return self.otype < other.otype

    def act(self):
        self.action(*self.args, **self.kwargs)
        
class OccurenceMonitor():
    def __init__(self):
        self.occurence_queue = []
        self.now = 0
        self.cnt =datetime.now()-datetime.now()
        
    def reset(self):
        self.occurence_queue = []
        self.now = 0
        self.cnt =datetime.now()-datetime.now()

    def add_occurence(self, occurence):
        heapq.heappush(self.occurence_queue, occurence)

    def step(self):
        tt=datetime.now()
        while self.occurence_queue and self.occurence_queue[0].trigger_time <= self.now:
            if self.occurence_queue[0].trigger_time < self.now:
                raise AssertionError('Found occurence ingored or not triggered in simulation, check the simulation code...')
            else:
                next_occurence = heapq.heappop(self.occurence_queue)
                next_occurence.act()
        self.cnt = self.cnt + datetime.now() - tt
        self.now = self.now + 1
        #if self.now % 1000 == 0:
        #    print('now time', self.now)

    def check_submit_occurence_only(self):
        is_submitting = False
        while (self.occurence_queue) and (self.occurence_queue[0].trigger_time <= self.now) and (self.occurence_queue[0].otype == Occurence.ARRIVAL):
            if self.occurence_queue[0].trigger_time < self.now:
                raise AssertionError('Found occurence ingored or not triggered in simulation, check the simulation code...')
            else:
                next_occurence = heapq.heappop(self.occurence_queue)
                next_occurence.act()     
            is_submitting = True 
        return True        
        