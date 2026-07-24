import os
import json
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from bot.robot_env_free_cont import TurtleBotEnv


class EpisodeCallback(BaseCallback):
    def __init__(self, max_episodes=1500, stats_file=None, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.stats_file = stats_file
        self.episodes = 0
        self.successes = 0
        self.collisions = 0

        # RIPRESA AUTOMATICA STATISTICHE: Carica lo storico reale se esiste
        if self.stats_file and os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r") as f:
                    data = json.load(f)
                    self.episodes = data.get("episodes", 0)
                    self.successes = data.get("successes", 0)
                    self.collisions = data.get("collisions", 0)
                    print(f"\n[MEMORIA DOCKER] 💾 Storico recuperato! Riparto dall'episodio {self.episodes}")
                    print(f"[MEMORIA DOCKER] Successi passati: {self.successes} | Collisioni passate: {self.collisions}\n")
            except Exception as e:
                print(f"Errore lettura file statistiche JSON: {e}")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue

            self.episodes += 1
            self.successes += int(info.get("episode_success", 0))
            self.collisions += int(info.get("collision", 0))

            success_rate = self.successes / self.episodes if self.episodes > 0 else 0.0
            collision_rate = self.collisions / self.episodes if self.episodes > 0 else 0.0

            # Registrazione su TensorBoard 
            self.logger.record("episodes/count", self.episodes)
            self.logger.record("episodes/success_rate", success_rate)
            self.logger.record("episodes/collision_rate", collision_rate)

            print(
                f"Episodio {self.episodes}/{self.max_episodes} | "
                f"success_rate={success_rate:.2f} | "
                f"collision_rate={collision_rate:.2f}"
            )

            # SALVATAGGIO IN TEMPO REALE: Aggiorna il file JSON ad ogni fine episodio
            if self.stats_file:
                try:
                    with open(self.stats_file, "w") as f:
                        json.dump({
                            "episodes": self.episodes,
                            "successes": self.successes,
                            "collisions": self.collisions
                        }, f)
                except Exception:
                    pass

            if self.episodes >= self.max_episodes:
                print(f"Stop: Raggiunto il limite di {self.episodes} episodi.")
                return False

        return True


def find_latest_checkpoint(models_dir):
    if not os.path.exists(models_dir):
        return None
   
    checkpoints = [
        f for f in os.listdir(models_dir)
        if f.startswith("turtlebot_free_cont") and f.endswith(".zip")
    ]

    if not checkpoints:
        return None

    # Ordina i file in base alla data di ultima modifica e prende il più recente
    return max(
        checkpoints,
        key=lambda f: os.path.getmtime(os.path.join(models_dir, f))
    )


def main():
    rclpy.init()

    print("Inizializzazione ambiente PPO Continuo...")

    models_dir = "./modelli_salvati_free_cont/"
    tensorboard_dir = "./ppo_tensorboard/"  # Porta standard Docker 6006

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # File JSON per salvare l'indice reale degli episodi
    stats_file = os.path.join(models_dir, "training_stats.json")

    env = Monitor(
        TurtleBotEnv(),
        info_keywords=("is_success", "episode_success", "collision", "timeout")
    )

    # Salva un checkpoint di sicurezza ogni 15.000 step
    checkpoint_callback = CheckpointCallback(
        save_freq=15000,
        save_path=models_dir,
        name_prefix="turtlebot_free_cont"
    )

    # Inizializza il callback per il conteggio degli episodi
    episode_callback = EpisodeCallback(max_episodes=1500, stats_file=stats_file)

    callbacks = CallbackList([checkpoint_callback, episode_callback])

    
    ultimo_checkpoint = find_latest_checkpoint(models_dir)

    if ultimo_checkpoint is None:
        print("\n[START] Nessun checkpoint trovato. Partenza da TABULA RASA (Modello Free Continuo).\n")
        
        
        if os.path.exists(stats_file):
            os.remove(stats_file)

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[128, 128])
        )
        reset_num_timesteps = True

    else:
        checkpoint_path = os.path.join(models_dir, ultimo_checkpoint)
        print(f"\n[RESUME] Trovato checkpoint! Ripresa dall'ultimo file di step: {checkpoint_path}\n")

        
        model = PPO.load(checkpoint_path, env=env)
        model.tensorboard_log = tensorboard_dir
        
        
        reset_num_timesteps = False

    print(f"Avvio addestramento: Target impostato a 1500 episodi complessivi.")

    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="PPO_Free_Cont"
        )

        final_path = os.path.join(models_dir, "turtlebot_free_cont_FINALE")
        model.save(final_path)
        print(f"Addestramento COMPLETATO! Modello finale salvato in: {final_path}")

    finally:
       
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()