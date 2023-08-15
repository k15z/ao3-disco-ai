# scp -r -P 40887 ao3-disco-ai root@142.126.60.45:ao3-disco-ai

# Test impact of hash size
python ao3_disco_ai/cli.py --experiment-name tune_hash_size --max-hash-size 1000
python ao3_disco_ai/cli.py --experiment-name tune_hash_size --max-hash-size 10000
python ao3_disco_ai/cli.py --experiment-name tune_hash_size --max-hash-size 100000

# Test impact of interactions
python ao3_disco_ai/cli.py --experiment-name test_interactions --no-use-interactions
python ao3_disco_ai/cli.py --experiment-name test_interactions --use-interactions
python ao3_disco_ai/cli.py --experiment-name test_interactions --max-hash-size 100000 --no-use-interactions
python ao3_disco_ai/cli.py --experiment-name test_interactions --max-hash-size 100000 --use-interactions

# Test impact of similarity loss
python ao3_disco_ai/cli.py --experiment-name test_interactions --similarity-loss-scale 0.1
python ao3_disco_ai/cli.py --experiment-name test_interactions --similarity-loss-scale 1.0
python ao3_disco_ai/cli.py --experiment-name test_interactions --similarity-loss-scale 10.0
