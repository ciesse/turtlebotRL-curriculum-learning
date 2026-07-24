import rclpy
from stable_baselines3 import PPO
import time


from bot.robot_env_forest import TurtleBotEnv

def main():
    rclpy.init()

    print(" Inizializzazione ambiente di test")
    env = TurtleBotEnv()

    # Percorso del modello finale addestrato nella fase 3
    model_path = "./modelli_salvati_forest/turtlebot_forest_INTERRUPTED.zip"
    
    print(f"Caricamento modello: {model_path}")
    try:
        model = PPO.load(model_path, env=env)
    except FileNotFoundError:
        print(f"ERRORE: Modello non trovato in {model_path}")
        return

    
    num_episodes = 100
    successi = 0
    collisioni = 0
    timeout = 0
    fallimenti_sconosciuti = 0

    print("\nAVVIO TEST DETERMINISTICO")
    print("-" * 30)

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done:
            deterministic=True
            action, _states = model.predict(obs, deterministic=True)
            
            # Il robot esegue l'azione
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            done = terminated or truncated

        # Analizza il risultato dell'episodio
        is_success = info.get("is_success", False)
        is_collision = info.get("collision", 0) == 1
        is_timeout = info.get("timeout", 0) == 1
        
        if is_success:
            successi += 1
            esito = "GOAL RAGGIUNTO"
        elif is_collision:
            collisioni += 1
            esito = "FALLITO PER COLLISIONE"
        elif is_timeout:
            timeout += 1
            esito = "FALLITO PER TIMEOUT"
        else:
            fallimenti_sconosciuti += 1
            esito = "FALLITO PER MOTIVO NON RICONOSCIUTO"
            
        print(f"Episodio {ep}/{num_episodes} | Passi: {step_count} | Esito: {esito}")
        
        
        time.sleep(1.0)

    # Statistiche finali
    success_rate = (successi / num_episodes) * 100
    collision_rate = (collisioni / num_episodes) * 100
    timeout_rate = (timeout / num_episodes) * 100
    unknown_rate = (fallimenti_sconosciuti / num_episodes) * 100

    print("-" * 30)
    print("RISULTATI FINALI DEL TEST")
    print(f"Successi: {successi} su {num_episodes}")
    print(f"Collisioni: {collisioni} su {num_episodes}")
    print(f"Timeout: {timeout} su {num_episodes}")
    
    if fallimenti_sconosciuti > 0:
        print(f"Fallimenti sconosciuti: {fallimenti_sconosciuti} su {num_episodes}")

    print("-" * 30)
    print(f"Success Rate Reale: {success_rate:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}%")
    print(f"Timeout Rate: {timeout_rate:.1f}%")

    if fallimenti_sconosciuti > 0:
        print(f"Unknown Failure Rate: {unknown_rate:.1f}%")

    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()