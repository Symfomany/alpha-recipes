"""
recipes/app/prompts.py

Prompts réutilisables pour le compagnon de recettes :
- analyse de la demande
- classification RAG (Adaptive)
- grading / réécriture (Corrective)
- agent de recettes (Agentic)
- batch cooking & étapes
"""


ANALYZE_REQUEST_PROMPT = """\
Tu es un assistant culinaire.

Analyse la demande utilisateur et retourne un JSON STRICT avec les clés :
- normalized_request : reformulation courte de la demande
- people : nombre de personnes (int, ou null)
- max_time_minutes : temps total maximum (int, ou null)
- diet : régime éventuel (vegan, végétarien, sans lactose, sans gluten, etc., ou null)
- allergies : liste de mots (ex: ["arachides", "lactose"])
- equipment_available : liste d'équipements (ex: ["four", "plaques", "mixeur"])

Demande utilisateur :
{query}
"""


CLASSIFY_RAG_PROMPT = """\
Tu dois décider de la meilleure stratégie pour récupérer des informations culinaires.

Réponds UNIQUEMENT par un des tokens suivants (sans texte en plus) :
- NO_RAG       : connaissance générale suffisante
- LOCAL_RECIPES: recettes locales (vector store recettes)
- COOKBOOKS    : PDF / livres de cuisine (vector store cookbooks)
- WEB          : recherche web (Tavily)

Question :
{query}
"""


GRADE_RETRIEVAL_PROMPT = """\
Tu es un évaluateur de RAG pour la cuisine.

On te donne une question et les documents récupérés (recettes, extraits de livres, résultats web).
Indique si ces documents sont :
- GOOD       : suffisants pour répondre précisément et en sécurité,
- BAD        : très insuffisants ou hors sujet,
- AMBIGUOUS  : partiellement utiles mais avec des zones d'ombre importantes.

Réponds UNIQUEMENT par GOOD, BAD ou AMBIGUOUS.

Question :
{query}

Documents :
{docs}
"""


REWRITE_QUERY_PROMPT = """\
Réécris la question suivante pour qu'elle soit plus précise et exploitable
par un moteur de recherche de recettes (RAG).

Conserve le français, clarifie les contraintes (temps, nombre de personnes, régime)
sans inventer de nouveaux éléments.

Question d'origine :
{query}
"""


CLARIFY_USER_PROMPT = """\
La question de cuisine suivante est ambiguë.

Formule UNE seule question courte pour demander les précisions les plus importantes
(régime, matériel disponible, budget, temps, niveau de cuisine, etc.).

Question utilisateur :
{query}
"""


AGENT_RECIPES_PROMPT = """\
Tu es un chef assistant.

À partir de la question de l'utilisateur et du contexte fourni (extraits de recettes,
techniques, résultats de recherche), propose 3 RECETTES CANDIDATES adaptées.

Pour chaque recette, donne :
- un titre,
- un résumé,
- une liste d'ingrédients (liste à puces),
- un temps total approximatif (en minutes),
- un niveau de difficulté (débutant / intermédiaire / avancé).

Réponds en français, sous forme de liste numérotée.

Question utilisateur :
{query}

Contexte disponible :
{context}
"""


BATCH_COOKING_PROMPT = """\
Tu es un chef spécialisé en batch cooking.

À partir des recettes candidates suivantes, construis un PLAN DE CUISINE optimisé
pour préparer plusieurs plats en un minimum de temps, en factorisant les tâches
(préparations communes, cuisson de grandes quantités, etc.).

Pour chaque étape, indique :
- les actions concrètes,
- les recettes concernées,
- le temps estimé,
- si possible les actions pouvant être faites en parallèle.

Demande utilisateur :
{query}

Recettes candidates :
{candidates}
"""


GENERATE_STEPS_PROMPT = """\
Tu es un chef pédagogue.

À partir du plan de batch cooking ci-dessous, génère des ÉTAPES DÉTAILLÉES de cuisson
pour l'utilisateur, en numérotant chaque étape et en indiquant le temps estimé.

Inclue :
- ordre conseillé,
- astuces de timing (préparer X pendant que Y cuit),
- rappels de sécurité (températures, cuisson viande/poisson),
- conseils de conservation (frigo / congélateur) si pertinent.

Demande utilisateur :
{query}

Plan de batch cooking :
{plan}
"""
