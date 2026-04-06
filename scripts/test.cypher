MATCH (p:Paper)<-[c:CITES]-()
WITH p, count(c) AS citations
ORDER BY citations DESC LIMIT 1
MATCH path = (citing:Paper)-[:CITES]->(p)<-[:CITES]-(other:Paper)
RETURN path LIMIT 50