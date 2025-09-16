use numpy::PyArray3;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::Cursor;
use tetanes_core::input::JoypadBtnState;
use tetanes_core::mem::Read;
use tetanes_core::prelude::*;
use tetanes_core::video::VideoFilter;

/// NES Emulator Environment for Reinforcement Learning
#[pyclass]
pub struct NesEnv {
    control_deck: ControlDeck,
    rom_loaded: bool,
}

#[pymethods]
#[allow(non_local_definitions)]
impl NesEnv {
    #[new]
    #[pyo3(signature = (headless = false))]
    fn new(headless: bool) -> Self {
        let mut config = Config::default();

        // Always use optimized settings for RL
        config.filter = VideoFilter::Pixellate;

        if headless {
            config.headless_mode = HeadlessMode::NO_AUDIO | HeadlessMode::NO_VIDEO;
        }

        let control_deck = ControlDeck::with_config(config);

        Self {
            control_deck,
            rom_loaded: false,
        }
    }

    /// Load a ROM from bytes
    fn load_rom(&mut self, rom_name: String, rom_data: &PyBytes) -> PyResult<()> {
        let mut cursor = Cursor::new(rom_data.as_bytes());
        self.control_deck
            .load_rom(rom_name, &mut cursor)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load ROM: {e}"
                ))
            })?;

        self.rom_loaded = true;
        self.reset()?;
        Ok(())
    }

    /// Reset the environment to initial state
    fn reset(&mut self) -> PyResult<()> {
        if !self.rom_loaded {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No ROM loaded",
            ));
        }

        self.control_deck.reset(ResetKind::Hard);
        Ok(())
    }

    /// Step the environment with given actions
    /// Actions: [player1_a, player1_b, player1_select, player1_start, player1_up, player1_down, player1_left, player1_right]
    #[pyo3(signature = (actions, render = true))]
    fn step(
        &mut self,
        actions: Vec<bool>,
        render: bool,
    ) -> PyResult<(PyObject, f64, bool, bool, PyObject)> {
        if !self.rom_loaded {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No ROM loaded",
            ));
        }

        // Validate action vector length
        if actions.len() != 8 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 8 actions, got {}",
                actions.len()
            )));
        }

        // Apply input actions
        self.apply_actions(&actions);

        // Step one frame
        let cycles = self.control_deck.clock_frame().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Emulation error: {e}"))
        })?;

        // Get observation
        let observation = if render {
            self.get_observation()?
        } else {
            Python::with_gil(|py| py.None())
        };

        // Placeholder values - SMB2 gym doesn't use these
        let reward = 0.0;
        let terminated = false;
        let truncated = false;

        // Info dict with cycle count
        let info = Python::with_gil(|py| {
            let info_dict = pyo3::types::PyDict::new(py);
            let _ = info_dict.set_item("cycles", cycles);
            info_dict.to_object(py)
        });

        Ok((observation, reward, terminated, truncated, info))
    }

    /// Get current frame as RGB array
    fn get_observation(&mut self) -> PyResult<PyObject> {
        let frame_buffer = self.control_deck.frame_buffer();

        Python::with_gil(|py| {
            // Create 3D array directly without intermediate allocation
            let mut reshaped = vec![vec![vec![0u8; 3]; 256]; 240];

            // Convert RGBA to RGB and reshape in one pass
            for (i, chunk) in frame_buffer.chunks_exact(4).enumerate() {
                let y = i / 256;
                let x = i % 256;
                if y < 240 {
                    reshaped[y][x][0] = chunk[0]; // R
                    reshaped[y][x][1] = chunk[1]; // G
                    reshaped[y][x][2] = chunk[2]; // B
                }
            }

            let array = PyArray3::<u8>::from_vec3(py, &reshaped)?;
            Ok(array.to_object(py))
        })
    }

    /// Save state to slot
    fn save_state(&mut self, slot: u8) -> PyResult<()> {
        let path = format!("save_state_{slot}.sav");
        self.control_deck.save_state(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to save state: {e}"))
        })
    }

    /// Load state from slot
    fn load_state(&mut self, slot: u8) -> PyResult<()> {
        let path = format!("save_state_{slot}.sav");
        self.control_deck.load_state(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load state: {e}"))
        })
    }

    /// Save state to a specific file path
    fn save_state_to_path(&mut self, path: &str) -> PyResult<()> {
        self.control_deck.save_state(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to save state to path '{path}': {e}"
            ))
        })
    }

    /// Load state from a specific file path
    fn load_state_from_path(&mut self, path: &str) -> PyResult<()> {
        self.control_deck.load_state(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load state from path '{path}': {e}"
            ))
        })
    }

    /// Read a single byte from RAM
    fn read_ram(&self, address: u16) -> PyResult<u8> {
        if address >= 0x800 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "RAM address {address:#x} out of bounds (must be < 0x800)"
            )));
        }

        Ok(self.control_deck.cpu().peek(address))
    }

    /// Set the frame speed for faster/slower emulation
    fn set_frame_speed(&mut self, speed: f32) -> PyResult<()> {
        if speed <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Frame speed must be positive",
            ));
        }

        self.control_deck.set_frame_speed(speed);
        Ok(())
    }

    /// Get the current frame speed
    fn get_frame_speed(&self) -> f32 {
        self.control_deck.frame_speed()
    }

    /// Get raw frame buffer as u16 array (240x256) without filtering
    fn get_raw_frame_buffer(&mut self) -> PyResult<PyObject> {
        let frame_buffer = self.control_deck.frame_buffer_raw();

        Python::with_gil(|py| {
            // Create 2D array of u16 values
            let mut reshaped = vec![vec![0u16; 256]; 240];

            // Copy raw PPU buffer
            for (i, &pixel) in frame_buffer.iter().enumerate() {
                let y = i / 256;
                let x = i % 256;
                if y < 240 {
                    reshaped[y][x] = pixel;
                }
            }

            // Convert to numpy array
            use numpy::PyArray2;
            let array = PyArray2::<u16>::from_vec2(py, &reshaped)?;
            Ok(array.to_object(py))
        })
    }

    /// Get grayscale frame as u8 array (240x256) - fastest method for RL
    fn get_grayscale_frame(&mut self) -> PyResult<PyObject> {
        let frame_buffer = self.control_deck.frame_buffer_raw();

        // NES palette grayscale lookup table (approximate luminance values)
        const GRAYSCALE_LUT: [u8; 64] = [
            84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 0, 0, 0, 152, 152, 152, 152, 152,
            152, 152, 152, 152, 152, 152, 152, 152, 0, 0, 0, 220, 220, 220, 220, 220, 220, 220,
            220, 220, 220, 220, 220, 220, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 0, 0, 0,
        ];

        Python::with_gil(|py| {
            // Create 2D array of u8 grayscale values
            let mut reshaped = vec![vec![0u8; 256]; 240];

            // Convert palette indices to grayscale in one pass
            for (i, &pixel_idx) in frame_buffer.iter().enumerate() {
                let y = i / 256;
                let x = i % 256;
                if y < 240 {
                    reshaped[y][x] = GRAYSCALE_LUT[pixel_idx as usize & 0x3F];
                }
            }

            use numpy::PyArray2;
            let array = PyArray2::<u8>::from_vec2(py, &reshaped)?;
            Ok(array.to_object(py))
        })
    }

    /// Get RGB frame using lookup table - faster than current get_observation
    fn get_rgb_frame(&mut self) -> PyResult<PyObject> {
        let frame_buffer = self.control_deck.frame_buffer_raw();

        // NES palette RGB lookup table (standard NTSC palette)
        const RGB_LUT: [(u8, u8, u8); 64] = [
            (84, 84, 84),
            (0, 30, 116),
            (8, 16, 144),
            (48, 0, 136),
            (68, 0, 100),
            (92, 0, 48),
            (84, 4, 0),
            (60, 24, 0),
            (32, 42, 0),
            (8, 58, 0),
            (0, 64, 0),
            (0, 60, 0),
            (0, 50, 60),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (152, 150, 152),
            (8, 76, 196),
            (48, 50, 236),
            (92, 30, 228),
            (136, 20, 176),
            (160, 20, 100),
            (152, 34, 32),
            (120, 60, 0),
            (84, 90, 0),
            (40, 114, 0),
            (8, 124, 0),
            (0, 118, 40),
            (0, 102, 120),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (236, 238, 236),
            (76, 154, 236),
            (120, 124, 236),
            (176, 98, 236),
            (228, 84, 236),
            (236, 88, 180),
            (236, 106, 100),
            (212, 136, 32),
            (160, 170, 0),
            (116, 196, 0),
            (76, 208, 32),
            (56, 204, 108),
            (56, 180, 204),
            (60, 60, 60),
            (0, 0, 0),
            (0, 0, 0),
            (236, 238, 236),
            (168, 204, 236),
            (188, 188, 236),
            (212, 178, 236),
            (236, 174, 236),
            (236, 174, 212),
            (236, 180, 176),
            (228, 196, 144),
            (204, 210, 120),
            (180, 222, 120),
            (168, 226, 144),
            (152, 226, 180),
            (160, 214, 228),
            (160, 162, 160),
            (0, 0, 0),
            (0, 0, 0),
        ];

        Python::with_gil(|py| {
            // Create 3D array of RGB values
            let mut reshaped = vec![vec![vec![0u8; 3]; 256]; 240];

            // Convert palette indices to RGB in one pass
            for (i, &pixel_idx) in frame_buffer.iter().enumerate() {
                let y = i / 256;
                let x = i % 256;
                if y < 240 {
                    let (r, g, b) = RGB_LUT[pixel_idx as usize & 0x3F];
                    reshaped[y][x][0] = r;
                    reshaped[y][x][1] = g;
                    reshaped[y][x][2] = b;
                }
            }

            let array = PyArray3::<u8>::from_vec3(py, &reshaped)?;
            Ok(array.to_object(py))
        })
    }
}

impl NesEnv {
    fn apply_actions(&mut self, actions: &[bool]) {
        debug_assert_eq!(
            actions.len(),
            8,
            "Actions should be validated before calling apply_actions"
        );

        let joypad = self.control_deck.joypad_mut(Player::One);

        // Map boolean actions to joypad buttons using const array for button states
        const BUTTON_STATES: [JoypadBtnState; 8] = [
            JoypadBtnState::A,
            JoypadBtnState::B,
            JoypadBtnState::SELECT,
            JoypadBtnState::START,
            JoypadBtnState::UP,
            JoypadBtnState::DOWN,
            JoypadBtnState::LEFT,
            JoypadBtnState::RIGHT,
        ];

        for (button, &pressed) in BUTTON_STATES.iter().zip(actions.iter()) {
            joypad.set_button(*button, pressed);
        }
    }
}

/// Python module for TetaNES RL environment
#[pymodule]
fn _tetanes(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<NesEnv>()?;
    Ok(())
}
