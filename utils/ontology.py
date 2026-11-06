"""GO ancestors and max-score propagation through is_a and same-aspect part_of."""
from pathlib import Path
from types import SimpleNamespace

NAMESPACES = {'molecular_function': 'mf', 'biological_process': 'bp', 'cellular_component': 'cc'}


class GeneOntology(dict):
    def __init__(self, path):
        super().__init__()
        records, record = [], None
        for line in Path(path).read_text().splitlines() + ['[End]']:
            if line.startswith('['):
                if record and 'id' in record:
                    records.append(record)
                record = {'parents': [], 'part_of': [], 'alt': []} if line == '[Term]' else None
            elif record is not None and ': ' in line:
                key, value = line.split(': ', 1)
                if key == 'is_a': record['parents'].append(value.split()[0])
                elif key == 'relationship' and value.startswith('part_of '): record['part_of'].append(value.split()[1])
                elif key == 'alt_id': record['alt'].append(value)
                elif key in ('id', 'name', 'namespace', 'is_obsolete'): record[key] = value
        for r in records:
            self[r['id']] = SimpleNamespace(id=r['id'], name=r['name'],
                namespace=NAMESPACES[r['namespace']], ns=NAMESPACES[r['namespace']], parents=set(r['parents']))
        for r in records:
            term = self[r['id']]
            term.parents.update(p for p in r['part_of'] if p in self and self[p].ns == term.ns)
            for alias in r['alt']:
                self[alias] = term
        self.root_term = ['GO:0008150', 'GO:0003674', 'GO:0005575']

    def ancestors(self, term):
        if not hasattr(self, '_ancestors'):
            self._ancestors = {}
        if term not in self._ancestors:
            found, todo = set(), [term]
            while todo:
                current = todo.pop()
                if current in self and current not in found:
                    found.add(current)
                    todo.extend(self[current].parents)
            self._ancestors[term] = found
        return self._ancestors[term]

    def transfer(self, terms):
        result = set()
        for term in terms:
            result.update(self.ancestors(term))
        return result

    def transfer_scores(self, scores):
        result = {}
        for term, score in scores.items():
            for ancestor in self.ancestors(term):
                result[ancestor] = max(result.get(ancestor, 0.), score)
        return result
