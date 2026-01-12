# Collaborative Robotics 2026

This repository contains software for controlling the **TidyBot2** mobile robot with **WX200** 6-DOF arm, developed for Professor Monroe Kennedy's 2026 Collaborative Robotics Class.

## Quick Start

### Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd collaborative-robotics-2026

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Run the Simulation

```bash
cd simulation/scripts
python drive_tidybot_wx200.py
```

This opens a MuJoCo viewer showing the TidyBot2 driving forward while waving its arm.

## Repository Structure

```
collaborative-robotics-2026/
├── simulation/          # MuJoCo simulation files
│   ├── scripts/         # Control and demo scripts
│   ├── assets/          # Robot models and meshes
│   └── README.md        # Simulation documentation
├── pyproject.toml       # Python project configuration
├── .python-version      # Python version specification
└── README.md            # This file
```

## Documentation

- **[Simulation Guide](simulation/README.md)**: Detailed guide for running and extending the MuJoCo simulation
- **Real Hardware Guide**: Coming soon

## Robot Specifications

**TidyBot2** is a mobile manipulation platform consisting of:
- **Mobile Base**: Kobuki or Create3 base with 3 DOF (x, y, theta)
- **WX200 Arm**: 5 DOF manipulator (550mm reach, 200g payload)
  - Waist (base rotation): ±175°
  - Shoulder (lift): -108° to 114°
  - Elbow (bend): -123° to 92°
  - Wrist angle: -100° to 123°
  - Wrist rotate: ±175°
- **Robotiq 2F-85 Gripper**: Adaptive parallel jaw gripper (85mm max opening)

## Development

### Adding New Features

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test in simulation first: `cd simulation/scripts && python your_script.py`
4. Create a pull request

### Running Tests

```bash
# Coming soon
uv run pytest
```

## Contributing

For course assignments and projects, follow the course guidelines for submission. For general contributions:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request

## Resources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [TidyBot2 Paper](https://arxiv.org/abs/2305.05658)
- [Interbotix WX200 Specs](https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/wx200.html)

## License

[Specify license]

## Contact

For questions about this repository or the course, contact [course staff contact info].
