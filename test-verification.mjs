import fetch from 'node-fetch';

async function testSimpleVerification() {
    try {
        console.log('Testing Marc verification...');
        
        const testText = "it, it, it honestly depends on the context...";
        
        const response = await fetch('http://localhost:3000/test_marc_simple', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: testText
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('\nTest Results:');
        console.log('Input text:', data.text);
        console.log('Is Marc:', data.is_marc);
        console.log('Success:', data.success);

        if (!data.success) {
            console.log('Error:', data.error);
        }

    } catch (error) {
        console.error('Error testing Marc verification:', error);
        console.error('Full error:', {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
    }
}

console.log('Starting verification test...');
testSimpleVerification();