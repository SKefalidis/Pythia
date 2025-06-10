import json

class GeospatialRelation:
    def __init__(self, function, argA, argB):
        self.function = function
        self.argA = argA
        self.argB = argB
        
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, GeospatialRelation):
            return False
        return check_matching_geospatial_relations(self, value)
    
    def __str__(self) -> str:
        return f"{self.function}({self.argA}, {self.argB})"
    
    def __hash__(self):
        return hash((self.function, self.argA, self.argB))
        
    @classmethod
    def from_llm_string(cls, relation_string: str):
        relation_string = relation_string.replace("<", "").replace(">", "")
        parts = relation_string.split(", ")
        argB = parts[1][:-1]
        function, argA = parts[0].split("(")[0], "".join(parts[0].split("(")[1:])
        argA = argA.strip()
        argB = argB.strip()
        return cls(function, argA, argB)
    
    @classmethod
    def from_gold_string(cls, relation_string: str):
        relation_string = relation_string.replace("_,", "_")
        relation_string = relation_string.replace(",_", "_")
        parts = relation_string.split("(")
        function = parts[0].strip()
        argA, argB = parts[1].rstrip(")").split(",")
        argA = argA.strip()
        argB = argB.strip()
        
        if function == "sfContains":
            function = "contains"
        elif function == "sfWithin":
            function = "contains"
            argA, argB = argB, argA
        elif function == "sfTouches":
            function = "touch"
        elif function == "sfOverlaps":
            function = "overlaps"
        elif function == "sfIntersects":
            function = "overlaps"
        elif function == "sfCrosses":
            function = "crosses"
        elif function == "above":
            function = "north_of"
        elif function == "below":
            function = "south_of"
        elif function == "left":
            function = "west_of"
        elif function == "right":
            function = "east_of"
        
        return cls(function, argA, argB)
    
def check_matching_geospatial_relations(gold_relation: GeospatialRelation, predicted_relation: GeospatialRelation) -> bool:
    # print(f"Checking relation: {gold_relation.function} vs {predicted_relation.function}")
    if gold_relation.function == "touch" and predicted_relation.function == "touch":
        if gold_relation.argA == predicted_relation.argB and gold_relation.argB == predicted_relation.argA:
            return True
    if gold_relation.function == "overlaps" and predicted_relation.function == "overlaps":
        if gold_relation.argA == predicted_relation.argB and gold_relation.argB == predicted_relation.argA:
            return True
    if gold_relation.function == "crosses" and predicted_relation.function == "crosses":
        if gold_relation.argA == predicted_relation.argB and gold_relation.argB == predicted_relation.argA:
            return True
    if gold_relation.function == "north_of" and predicted_relation.function == "south_of":
        return (gold_relation.argA == predicted_relation.argB and
                gold_relation.argB == predicted_relation.argA)
    if gold_relation.function == "south_of" and predicted_relation.function == "north_of":
        return (gold_relation.argA == predicted_relation.argB and
                gold_relation.argB == predicted_relation.argA)
    if gold_relation.function == "west_of" and predicted_relation.function == "east_of":
        return (gold_relation.argA == predicted_relation.argB and
                gold_relation.argB == predicted_relation.argA)
    if gold_relation.function == "east_of" and predicted_relation.function == "west_of":
        return (gold_relation.argA == predicted_relation.argB and
                gold_relation.argB == predicted_relation.argA)
    return (gold_relation.function == predicted_relation.function and
            gold_relation.argA == predicted_relation.argA and
            gold_relation.argB == predicted_relation.argB)
    
    

gold_relations = json.load(open("./geoq1089_gold_relations.json", "r"))
predicted_relations = json.load(open("./GeoQuestions1089_with_geospatial_relations_descriptors_distance_gpt_4_1_final.json", "r"))

processed = 0
skipped = 0

tp = 0
fp = 0
fn = 0

for idx in range(1, 894):
    skip = False
    entry = str(idx)
    if entry not in gold_relations or entry not in predicted_relations:
        print(f"Skipping entry {entry} as it is not present in both datasets.")
        continue
    converted_gold_relations = []
    converted_predicted_relations = []
    for gold_relation in gold_relations[entry]["relationships"]:
        if "(_" in gold_relation or "_)" in gold_relation:
            skip = True
            break
        processed += 1
        gold_relation = GeospatialRelation.from_gold_string(gold_relation)
        converted_gold_relations.append(gold_relation)
        
    if skip:
        skipped += 1
        continue
        
    for predicted_relation in predicted_relations[entry]["geospatial_relations"]:
        # print(entry)
        predicted_relation = GeospatialRelation.from_llm_string(predicted_relation)
        converted_predicted_relations.append(predicted_relation)
    
    local_tp = 0
    local_fp = 0
    local_fn = 0
    for predicted_relation in converted_predicted_relations:
        matched = False
        for gold_relation in converted_gold_relations:
            if check_matching_geospatial_relations(gold_relation, predicted_relation):
                matched = True
                break
        if matched:
            local_tp += 1
        else:
            local_fp += 1
    
    tp += local_tp
    fp += local_fp
    fn += local_fn
    
    if local_fp > 0:
        print(f"Entry {entry} has false positives: {fp} (TP: {tp}, FN: {fn})")
        print(f"Predicted: {[str(rel) for rel in converted_predicted_relations]}")
        print(f"Gold: {[str(rel) for rel in converted_gold_relations]}")
        # exit(0)
    
    if local_fn > 0:
        print(f"Entry {entry} has false negatives: {fn} (TP: {tp}, FP: {fp})")
        print(f"Predicted: {[str(rel) for rel in converted_predicted_relations]}")
        print(f"Gold: {[str(rel) for rel in converted_gold_relations]}")
        # exit(0)
    
print(f"Processed {processed} relations.")
print(f"Skipped {skipped} relations due to invalid format.")

print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")