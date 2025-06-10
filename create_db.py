from neo4j import GraphDatabase

# ---- Configuration ----
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = ""  # 🔁 Replace with your actual password



# Add the synthea dataset in the  Neo4j Database Imort Directory
# Execute the Script to populate the database with the Synthea dataset 




# ---- Full Cypher Script ----
cypher_script = """
CREATE INDEX patient_id_index IF NOT EXISTS FOR (p:Patient) ON (p.Id);
LOAD CSV WITH HEADERS FROM 'file:///patients.csv' AS row
MERGE (p:Patient {Id: row.Id})
SET p.BIRTHDATE = row.BIRTHDATE,
    p.DEATHDATE = row.DEATHDATE,
    p.SSN = row.SSN,
    p.DRIVERS = row.DRIVERS,
    p.PASSPORT = row.PASSPORT,
    p.PREFIX = row.PREFIX,
    p.FIRST = row.FIRST,
    p.LAST = row.LAST,
    p.SUFFIX = row.SUFFIX,
    p.MAIDEN = row.MAIDEN,
    p.MARITAL = row.MARITAL,
    p.RACE = row.RACE,
    p.ETHNICITY = row.ETHNICITY,
    p.GENDER = row.GENDER,
    p.BIRTHPLACE = row.BIRTHPLACE,
    p.ADDRESS = row.ADDRESS;

CREATE INDEX encounter_id_index IF NOT EXISTS FOR (e:Encounter) ON (e.Id);
LOAD CSV WITH HEADERS FROM 'file:///encounters.csv' AS ROW
MERGE (e:Encounter {Id: ROW.Id})
SET e.START = ROW.START,
    e.STOP = ROW.STOP,
    e.PATIENT = ROW.PATIENT,
    e.CODE = ROW.CODE,
    e.DESCRIPTION = ROW.DESCRIPTION,
    e.REASONCODE = ROW.REASONCODE,
    e.REASONDESCRIPTION = ROW.REASONDESCRIPTION;

MATCH (p:Patient), (e:Encounter)
WHERE p.Id = e.PATIENT
CREATE (p)-[:HAS_ENCOUNTER]->(e);

CREATE INDEX observation_id_index IF NOT EXISTS FOR (o:Observation) ON (o.Id);
LOAD CSV WITH HEADERS FROM 'file:///observations.csv' AS row
WITH row, apoc.create.uuid() AS uuid
CREATE (o:Observation {
    Id: uuid,
    DATE: row.DATE,
    PATIENT: row.PATIENT,
    ENCOUNTER: row.ENCOUNTER,
    CODE: row.CODE,
    DESCRIPTION: row.DESCRIPTION,
    VALUE: row.VALUE,
    UNITS: row.UNITS
});
MATCH (p:Patient), (o:Observation)
WHERE p.Id = o.PATIENT
MERGE (p)-[:HAS_OBSERVATION]->(o);
MATCH (e:Encounter), (o:Observation)
WHERE e.Id = o.ENCOUNTER
MERGE (o)-[:PART_OF_ENCOUNTER]->(e);

CREATE INDEX medication_id_index IF NOT EXISTS FOR (m:Medication) ON (m.Id);
LOAD CSV WITH HEADERS FROM 'file:///medications.csv' AS row
WITH row, apoc.create.uuid() AS uuid
CREATE (m:Medication {
    Id: uuid,
    START: row.START,
    STOP: row.STOP,
    PATIENT: row.PATIENT,
    ENCOUNTER: row.ENCOUNTER,
    CODE: row.CODE,
    DESCRIPTION: row.DESCRIPTION,
    REASONCODE: row.REASONCODE,
    REASONDESCRIPTION: row.REASONDESCRIPTION
});
MATCH (P:Patient), (M:Medication)
WHERE P.Id = M.PATIENT
MERGE (P)-[:HAS_MEDICATION]->(M);
MATCH (E:Encounter), (M:Medication)
WHERE E.Id = M.ENCOUNTER
MERGE (M)-[:PART_OF_ENCOUNTER]->(E);

CREATE INDEX procedure_id_index IF NOT EXISTS FOR (p:Procedure) ON (p.Id);
LOAD CSV WITH HEADERS FROM 'file:///procedures.csv' AS row
WITH row, apoc.create.uuid() AS uuid
CREATE (p:Procedure {
    Id: uuid,
    START: row.START,
    STOP: row.STOP,
    PATIENT: row.PATIENT,
    ENCOUNTER: row.ENCOUNTER,
    CODE: row.CODE,
    DESCRIPTION: row.DESCRIPTION,
    REASONCODE: row.REASONCODE,
    REASONDESCRIPTION: row.REASONDESCRIPTION
});
MATCH (po:Procedure), (p:Patient)
WHERE p.Id = po.PATIENT
MERGE (p)-[:HAS_PROCEDURE]->(po);
MATCH (e:Encounter), (po:Procedure)
WHERE e.Id = po.ENCOUNTER
MERGE (po)-[:PART_OF_ENCOUNTER]->(e);

CREATE INDEX condition_id_index IF NOT EXISTS FOR (c:Condition) ON (c.Id);
LOAD CSV WITH HEADERS FROM 'file:///conditions.csv' AS row
WITH row, apoc.create.uuid() AS uuid
CREATE (c:Condition {
    Id: uuid,
    START: row.START,
    STOP: row.STOP,
    PATIENT: row.PATIENT,
    ENCOUNTER: row.ENCOUNTER,
    CODE: row.CODE,
    DESCRIPTION: row.DESCRIPTION
});
MATCH (c:Condition), (p:Patient)
WHERE p.Id = c.PATIENT
CREATE (p)-[:HAS_CONDITION]->(c);
MATCH (e:Encounter), (c:Condition)
WHERE e.Id = c.ENCOUNTER
CREATE (c)-[:PART_OF_ENCOUNTER]->(e);

CREATE INDEX immunization_id_index IF NOT EXISTS FOR (i:Immunization) ON (i.Id);
LOAD CSV WITH HEADERS FROM 'file:///immunizations.csv' AS row
WITH row, apoc.create.uuid() AS uuid
CREATE (i:Immunization {
    Id: uuid,
    DATE: row.DATE,
    PATIENT: row.PATIENT,
    ENCOUNTER: row.ENCOUNTER,
    CODE: row.CODE,
    DESCRIPTION: row.DESCRIPTION
});
MATCH (i:Immunization), (p:Patient)
WHERE p.Id = i.PATIENT
CREATE (p)-[:HAS_IMMUNIZATION]->(i);
MATCH (e:Encounter), (c:Immunization)
WHERE e.Id = c.ENCOUNTER
CREATE (c)-[:PART_OF_ENCOUNTER]->(e);

CREATE INDEX allergy_id_index IF NOT EXISTS FOR (i:Allergy) ON (i.Id);
LOAD CSV WITH HEADERS FROM 'file:///allergies.csv' AS row
WITH row, apoc.create.uuid() AS uuid
CREATE (i:Allergy {
    Id: uuid,
    START: row.START,
    STOP: row.STOP,
    PATIENT: row.PATIENT,
    ENCOUNTER: row.ENCOUNTER,
    CODE: row.CODE,
    DESCRIPTION: row.DESCRIPTION
});
MATCH (i:Allergy), (p:Patient)
WHERE p.Id = i.PATIENT
CREATE (p)-[:HAS_ALLERGY]->(i);
MATCH (e:Encounter), (c:Allergy)
WHERE e.Id = c.ENCOUNTER
CREATE (c)-[:PART_OF_ENCOUNTER]->(e);

CREATE INDEX careplan_id_index IF NOT EXISTS FOR (m:Careplan) ON (m.Id);
LOAD CSV WITH HEADERS FROM 'file:///careplans.csv' AS row
CREATE (m:Careplan {
    Id: row.Id,
    START: row.START,
    STOP: row.STOP,
    PATIENT: row.PATIENT,
    ENCOUNTER: row.ENCOUNTER,
    CODE: row.CODE,
    DESCRIPTION: row.DESCRIPTION,
    REASONCODE: row.REASONCODE,
    REASONDESCRIPTION: row.REASONDESCRIPTION
});
MATCH (i:Careplan), (p:Patient)
WHERE p.Id = i.PATIENT
CREATE (p)-[:HAS_CAREPLAN]->(i);
MATCH (e:Encounter), (c:Careplan)
WHERE e.Id = c.ENCOUNTER
CREATE (c)-[:PART_OF_ENCOUNTER]->(e);
"""

# ---- Run the Cypher Script ----
def run_cypher_script(uri, user, password, script):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for query in script.split(";"):
            q = query.strip()
            if q:
                print(f"Executing:\n{q[:100]}...")
                session.run(q)
    driver.close()
    print("✅ All Cypher queries executed successfully.")

# ---- Execute ----
if __name__ == "__main__":
    run_cypher_script(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, cypher_script)
