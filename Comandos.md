1) carpeta del proyecto:
cd "C:\Users\rdpal\OneDrive\Documentos\Obsidian Vault\ObsidianRepo\Brain\5. Apps\Mini-proyecto Sentimientos"

2) git pull origin main

3) Dependencias:
pip install -r requirements.txt

4) El experimento completo: entrena y mide accuracy con las 3871 reseñas reales
python run_experiment.py --k 15

5) (opcional, para mostrar el barrido de k y el hallazgo Jaccard>Coseno)
python barrido_k.py

6) clasificar una reseña ahí mismo
python predict_review.py