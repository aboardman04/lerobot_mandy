from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    policy_path = "aboardman/act_so101_test11_policy"

    print(f"Loading policy: {policy_path}")
    # Actually let's use ACTPolicy.from_pretrained directly
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()

    print("Policy loaded successfully.")


if __name__ == "__main__":
    main()
