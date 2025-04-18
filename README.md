# SpotiRec

# Registra che un utente ha ascoltato un brano
curl -X POST http://localhost:5000/listened \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U10", "track_id": "T001"}'

# Ottieni una raccomandazione per un utente
curl http://localhost:5000/recommend/U10

# Ottieni lo storico degli ascolti di un utente
curl http://localhost:5000/history/U10

# Ottieni i brani più popolari
curl http://localhost:5000/popular