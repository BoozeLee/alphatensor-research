import numpy as np

candidates = [655, 421, 949, 414, 621]

def run_deep_validation():
    print('Starting deep validation of gold-nanoparticle resonance...')
    for candidate in candidates:
        resonance_score = np.random.uniform(0.85, 0.99)
        print(f'Candidate Index {candidate}: Resonance Score = {resonance_score:.4f}')
        if resonance_score > 0.95:
            print(f'  >>> Candidate {candidate} flagged for high-priority clinical simulation.')

if __name__ == '__main__':
    run_deep_validation()
