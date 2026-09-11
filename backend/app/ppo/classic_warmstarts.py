"""Disclosed, training-only assistance for sparse classic-control rewards."""
from ..envs import mountain_car
from .demonstrations import BehaviorCloningWarmStart
from .initialization import ActorInitialization


MOUNTAIN_CAR_ACTOR_INITIALIZATION = ActorInitialization(
    scope="mountain_car_demonstration_assisted_ppo",
    continuous_action_labels=("motor_force",),
    continuous_action_prior=(0.0,),
    continuous_log_std=(-2.0,),
)

MOUNTAIN_CAR_ACTOR_WARM_START = BehaviorCloningWarmStart(
    id="mountain-car-momentum-demonstrations-v1",
    expert_id="mountain-car-momentum-sign-v1",
    expert_description=(
        "training-only momentum pumping: full positive motor command for "
        "positive velocity, full negative command otherwise; the neural "
        "actor acts alone during PPO and evaluation"),
    dataset_seed_base=4_100_000,
    dataset_episodes=80,
    dataset_start_description=(
        "ordinary jittered canonical valley starts; demonstrations use "
        "seeds disjoint from selection and validation"),
    dataset_builder=mountain_car.behavior_cloning_dataset,
    continuous_action_labels=("motor_force",),
    continuous_loss_weights=(1.0,),
    epochs=60,
)
