#! /usr/bin/python
class ParetoData:
    def __init__(self):
        self._pareto_solutions = {}
        self._configs = {}
        self._cnt = 0

    def update_pareto_solutions(self, sample, exec_time, max_rss):
        new_item = True
        if self._pareto_solutions:
            for key in list(self._pareto_solutions):
                if self._pareto_solutions[key][0] < exec_time and self._pareto_solutions[key][1] < max_rss:
                    new_item = False
                    break
                elif self._pareto_solutions[key][0] > exec_time and self._pareto_solutions[key][1] > max_rss:
                    self._pareto_solutions[key] = [exec_time, max_rss]
                    self._configs[key] = sample
                    new_item = False

        if new_item is True:
            self._pareto_solutions[self._cnt] = [exec_time, max_rss]
            self._configs[self._cnt] = sample
            self._cnt += 1

    def dump_pareto_solutions(self, dumpdata):
        marked = {}
        pareto_results = []
        for i in range(self._cnt):
            if self._configs[i] not in marked:
                marked[self._configs[i]] = True
                pareto_results.append({
                    "id": self._configs[i],
                    "exec_time": self._pareto_solutions[i][0],
                    "max_rss": self._pareto_solutions[i][1]
                })
        dumpdata.update({"solutions": pareto_results})

        return dumpdata
