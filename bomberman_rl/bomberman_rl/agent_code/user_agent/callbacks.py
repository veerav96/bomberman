def setup(self):
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    self.logger.info(f'game_state before action {game_state}')
    return game_state['user_input']
