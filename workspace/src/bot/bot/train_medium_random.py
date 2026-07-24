import os
import json
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from bot.robot_env_medium_cont_random import TurtleBotEnv


class EpisodeCallback(BaseCallback):
    def __init__(self, max_episodes=5000, stats_file=None, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.stats_file = stats_file
        self.episodes = 0
        self.successes = 0
        self.collisions = 0
        self.timeouts = 0

        # Recupero delle statistiche precedenti se il file esiste
        if self.stats_file and os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r") as f:
                    data = json.load(f)
                    self.episodes = data.get("episodes", 0)
                    self.successes = data.get("successes", 0)
                    self.collisions = data.get("collisions", 0)
                    self.timeouts = data.get("timeouts", 0)  
                    print(f"\n[MEMORIA DOCKER] Storico recuperato. Riparto dall'episodio {self.episodes}")
                    print(f"[MEMORIA DOCKER] Successi: {self.successes} | Collisioni: {self.collisions} | Timeouts: {self.timeouts}\n")
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
            self.timeouts += int(info.get("timeout", 0))

            success_rate = self.successes / self.episodes if self.episodes > 0 else 0.0
            collision_rate = self.collisions / self.episodes if self.episodes > 0 else 0.0
            timeout_rate = self.timeouts / self.episodes if self.episodes > 0 else 0.0

            # Registrazione su TensorBoard
            self.logger.record("episodes/count", self.episodes)
            self.logger.record("episodes/success_rate", success_rate)
            self.logger.record("episodes/collision_rate", collision_rate)
            self.logger.record("episodes/timeout_rate", timeout_rate)

            print(
                f"Episodio {self.episodes}/{self.max_episodes} | "
                f"success_rate={success_rate:.2f} | "
                f"collision_rate={collision_rate:.2f} | "
                f"timeout_rate={timeout_rate:.2f}"
            )

            # Salvataggio statistiche in tempo reale
            if self.stats_file:
                try:
                    with open(self.stats_file, "w") as f:
                        json.dump({
                            "episodes": self.episodes,
                            "successes": self.successes,
                            "collisions": self.collisions,
                            "timeouts": self.timeouts
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
        if f.startswith("turtlebot_medium_cont_random") and f.endswith(".zip")
    ]

    if not checkpoints:
        return None

    return max(
        checkpoints,
        key=lambda f: os.path.getmtime(os.path.join(models_dir, f))
    )


def main():
    rclpy.init()

    print("Inizializzazione ambiente PPO Continuo (Medium Random Spawn)...")
    models_dir = "./modelli_salvati_medium_cont_random/"
    tensorboard_dir = "./ppo_tensorboard/"  

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # File JSON per le statistiche
    stats_file = os.path.join(models_dir, "training_stats_medium_random.json")

    env = Monitor(
        TurtleBotEnv(),
        info_keywords=("is_success", "episode_success", "collision", "timeout")
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=15000,
        save_path=models_dir,
        name_prefix="turtlebot_medium_cont_random"
    )

    episode_callback = EpisodeCallback(max_episodes=5000, stats_file=stats_file)

    callbacks = CallbackList([checkpoint_callback, episode_callback])

    ultimo_checkpoint = find_latest_checkpoint(models_dir)

    if ultimo_checkpoint is None:
        # TRANSFER LEARNING
        modello_base_path = "./modelli_salvati_free_cont/turtlebot_free_cont_FINALE.zip"
        
        if os.path.exists(modello_base_path):
            print(f"\n[TRANSFER LEARNING] Caricamento modello base: {modello_base_path}\n")
            model = PPO.load(modello_base_path, env=env)
            
            # Iniezione di entropia per esplorare gli spawn casuali
            model.ent_coef = 0.05 
            
            model.tensorboard_log = tensorboard_dir
            reset_num_timesteps = True 
        else:
            print("\n[ERRORE] Modello FINALE base non trovato. Inizializzazione nuovo modello da zero.\n")
            model = PPO(
                "MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_dir, 
                learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, 
                gamma=0.99, ent_coef=0.05, clip_range=0.2, policy_kwargs=dict(net_arch=[128, 128])
            )
            reset_num_timesteps = True

        if os.path.exists(stats_file):
            os.remove(stats_file)

    else:
        # RESUME
        checkpoint_path = os.path.join(models_dir, ultimo_checkpoint)
        print(f"\n[RESUME] Trovato checkpoint. Ripresa dall'ultimo file: {checkpoint_path}\n")

        model = PPO.load(checkpoint_path, env=env)
        
        # Entropia di mantenimento al riavvio
        model.ent_coef = 0.001 
        
        model.tensorboard_log = tensorboard_dir
        reset_num_timesteps = False

    print("Avvio addestramento: Target impostato a 5000 episodi.")

    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="PPO_Medium_Cont_Random"  
        )
        
        final_path = os.path.join(models_dir, "turtlebot_medium_cont_RANDOM_FINALE")
        model.save(final_path)
        print(f"Addestramento MEDIUM RANDOM COMPLETATO! Modello finale salvato in: {final_path}")

    finally:
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()